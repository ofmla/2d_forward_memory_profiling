#!/usr/bin/env python
# coding: utf-8

# basic imports.
import os
import sys
import segyio
import numpy as np
import time
import yaml
import json
import h5py
import gc
from timeit import default_timer as timer

from dask_jobqueue import SLURMCluster
import multiprocessing.popen_spawn_posix
from dask.distributed import Client, LocalCluster, wait
from dask.distributed import performance_report
from dask.distributed import as_completed, get_worker, progress
from distributed.worker import logger
from joblib import Parallel, delayed

from devito import *
from examples.checkpointing import DevitoCheckpoint, CheckpointOperator
from examples.seismic import SeismicModel, AcquisitionGeometry, TimeAxis
from examples.seismic import Receiver, PointSource
from examples.seismic.tti import AnisotropicWaveSolver
from examples.seismic.tti.operators import kernel_centered_2d
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.acoustic.operators import iso_stencil
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver

from pyrevolve import Revolver

from utils import segy_write, make_lookup_table, create_shot_dict
from utils import limit_model_to_receiver_area, extend_image, check_par_attr
from ctypes import c_float


class DaskCluster:
    "Class for using dask tasks to parallelize calculation of shots"

    def __init__(self):
        config_file = os.path.join(os.getcwd(), "config", "config.yaml")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), config_file)
        with open(config_file) as file:
            self.config_values = yaml.load(file, Loader=yaml.FullLoader)
        if "queue" not in self.config_values:
            self.config_values["queue"] = "GPUlongC"
        if "project" not in self.config_values:
            self.config_values["project"] = "cenpes-lde"
        if "n_workers" not in self.config_values:
            self.config_values["n_workers"] = 4
        if "cores" not in self.config_values:
            self.config_values["cores"] = 36
        if "processes" not in self.config_values:
            self.config_values["processes"] = 1
        if "memory" not in self.config_values:
            self.config_values["memory"] = 320
        if "job_extra" not in self.config_values:
            self.config_values["job_extra"] = ['-e slurm-%j.err', '-o slurm-%j.out',
                                               '--job-name="dask_task"']

        if self.config_values["use_local_cluster"]:
            # single-threaded execution, as this is actually best for the workload
            cluster = LocalCluster(n_workers=self.config_values["n_workers"],
                                   threads_per_worker=1,
                                   memory_limit='2.5GB', death_timeout=60,
                                   resources={'process': 1})
        else:
            cluster = SLURMCluster(queue=self.config_values["queue"],
                                   project=self.config_values["project"],
                                   cores=self.config_values["cores"],
                                   processes=self.config_values["processes"],
                                   memory=str(self.config_values["memory"])+"GB",
                                   death_timeout='60',
                                   interface='ib0',
                                   extra=['--resources "process=1"'],
                                   job_extra=self.config_values["job_extra"])

            # Scale cluster to n_workers
            cluster.scale(jobs=self.config_values["n_workers"])

        self.func_map = {'shots': DaskCluster.gen_shot_in_worker,
                         'grads': DaskCluster.gen_grad_in_worker}

        # Wait for cluster to start
        time.sleep(30)
        self.client = Client(cluster)
        # initialize tasks dictionary
        self._set_tasks_from_files()

    def _set_tasks_from_files(self):
        "Returns a dict which contains the list of tasks to be run"

        if self.config_values["forward"]:
            file_src = self.config_values['file_src']
            file_rec = self.config_values['file_rec']

            if file_src is not None and file_rec is not None:
                with h5py.File(file_src, 'r') as f:
                    a_group_key = list(f.keys())[0]
                    src_coordinates = f[a_group_key][()]
                nshots = src_coordinates.shape[0]

                with h5py.File(file_rec, 'r') as f:
                    a_group_key = list(f.keys())[0]
                    rec_coordinates = f[a_group_key][()]
            else:
                nshots = self.config_values['nshots']
                nrecs = self.config_values['nrecs']
                model_size = self.config_values['model_size']
                src_step = self.config_values['src_step']
                # Define acquisition geometry: receivers
                # First, sources position
                src_coord = np.empty((nshots, 2))
                src_coord[:, 0] = np.arange(start=0., stop=model_size, step=src_step)
                src_coord[:, -1] = self.config_values['src_depth']
                # Initialize receivers for synthetic and imaging data
                rec_coord = np.empty((nrecs, 2))
                rec_coord[:, 0] = np.linspace(0, model_size, num=nrecs)
                rec_coord[:, 1] = self.config_values['rec_depth']

            self.tasks_dict = {i: {'Source': src_coord[i],
                                   'Receivers': rec_coord} for i in range(nshots)}
        else:
            # Read chunk of shots
            segy_dir_files = self.config_values['solver_params']['shotfile_path']
            segy_files = [f for f in os.listdir(segy_dir_files) if f.endswith('.segy')]
            segy_files = [segy_dir_files + sub for sub in segy_files]

            # Create a dictionary of shots
            self.tasks_dict = {}
            for count, sfile in enumerate(segy_files, start=1):
                self.tasks_dict.update({str(count) if k == 1 else k: v
                                       for k, v in make_lookup_table(sfile).items()})

    def launch_tasks(self, func):

        shot_futures = []
        all_shot_results = []
        par = self.client.scatter({**self.config_values['solver_params'],
                                   **self.config_values['ckp_params']}, broadcast=True)

        shot_master_list = [(lambda d: d.update(id=key) or d)(val)
                            for (key, val) in self.tasks_dict.items()]
        index = 1
        batch = []
        for shot in shot_master_list:
            batch.append(shot)
            if index % self.config_values["shot_batch_size"] == 0:
                shot_futures = self.client.map(func,
                                               batch, solver_params=par,
                                               resources={'process': 1})
                all_shot_results.extend(self.client.gather(shot_futures))
                batch = []
                print('Completed shots till {0:3d}'.format(index))
                index = index + 1
        if batch:
            shot_futures = self.client.map(func,
                                           batch, solver_params=par,
                                           resources={'process': 1})
            all_shot_results.extend(self.client.gather(shot_futures))

        return all_shot_results

    def generate_shots_in_cluster(self):
        "Forward modeling for all the shots in parallel in a dask cluster"

        all_shot_results = self.launch_tasks(DaskCluster.gen_shot_in_worker)

        if all(all_shot_results):
            print("Successfully generated %d shots" % (len(all_shot_results)))
        else:
            raise Exception("Some error occurred. Please check logs")

    def generate_grad_in_cluster(self, X):
        "Gradient computing for all the shots in parallel in a dask cluster"

        start_time = time.time()
        self.client.restart()

        shape = self.config_values['solver_params']['shape']
        with h5py.File('vp_current.h5', 'w') as f:
            f.create_dataset('current_dataset', data=X.reshape(-1, shape[1]))

        if self.config_values['ckp_params']['checkpointing']:
            all_shot_results = self.launch_tasks(DaskCluster.gen_grad_in_worker_ckp)
        else:
            all_shot_results = self.launch_tasks(DaskCluster.gen_grad_in_worker)

        spacing = self.config_values['solver_params']['spacing']

        if len(shape) == 2:
            domain_size = ((shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1])
        else:
            domain_size = ((shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1],
                           (shape[2] - 1) * spacing[2])

        grid = Grid(shape=shape, extent=domain_size)
        f = Function(name='f', grid=grid)
        grad = Function(name='g', grid=grid)
        grad_update = Inc(grad, f)
        op_grad = Operator([grad_update])

        grad.data[:] = all_shot_results[0][1]
        objective = all_shot_results[0][0]
        i = 1

        # Iterating using while loop
        while i < len(all_shot_results):
            f.data[:] = all_shot_results[i][1]
            op_grad.apply()
            objective += all_shot_results[i][0]
            i += 1

        mute_depth = self.config_values['mute_depth']
        if mute_depth is not None:
            grad.data[:, 0:mute_depth] = 0.

        elapsed_time = time.time() - start_time
        print("Cost_fcn eval took {0:8.2f} sec - Cost_fcn={1:10.3E}".format(elapsed_time,
                                                                            objective))

        return c_float(objective), np.ravel(grad.data[:]).astype(np.float32)

    @staticmethod
    def gen_shot_in_worker(shot_dict, solver_params):

        dtype = solver_params['dtype']

        if dtype == 'float32':
            dtype = np.float32
        elif dtype == 'float64':
            dtype = np.float64
        else:
            raise ValueError("Invalid dtype")

        # Metadata from hdf5 file
        with h5py.File(solver_params['parfile_path']+'vp.h5', 'r') as f:
            metadata = json.loads(f['metadata'][()])

        origin = metadata['origin']
        shape = metadata['shape']
        spacing = metadata['spacing']
        setup_func = solver_params['setup_func']
        model_name = solver_params['model_name']

        #
        if setup_func == 'acoustic':
            vp = np.empty(shape)
            pars = ['vp']
            params = [vp]
        elif setup_func == 'tti':
            vp = np.empty(shape)
            epsilon = np.empty(shape)
            delta = np.empty(shape)
            theta = np.empty(shape)
            pars = ['vp', 'delta', 'epsilon', 'theta']
            params = [vp, delta, epsilon, theta]
            if len(shape) == 3:
                phi = np.empty(shape)
                pars.extend(['phi'])
                params.extend([phi])
        else:
            vp = np.empty(shape)
            qp = np.empty(shape)
            rho = np.empty(shape)
            pars = ['vp', 'qp', 'rho']
            params = [vp, qp, rho]

        # Read medium parameters
        for file, par in zip(pars, params):
            with h5py.File(solver_params['parfile_path']+file+'.h5', 'r') as f:
                par[:] = f[file][()]

        if setup_func == 'tti':
            theta *= (np.pi/180.)  # use radians

        # Get src/rec coords
        src_coord = np.array(shot_dict['Source']).reshape((1, len(shape)))
        rec_coord = np.array(shot_dict['Receivers'])

        space_order = solver_params['space_order']
        dt = solver_params['dt']
        nbl = solver_params['nbl']
        f0 = solver_params['f0']
        t0 = solver_params['t0']
        tn = solver_params['tn']

        params = None if setup_func == 'acoustic' else params[1:]
        model = limit_model_to_receiver_area(rec_coord, src_coord, origin, spacing,
                                             shape, vp, params, space_order,
                                             nbl, buffer=10)[0]

        if solver_params['born']:
            model0 = limit_model_to_receiver_area(rec_coord, src_coord, origin, spacing,
                                                  shape, vp, params, space_order,
                                                  nbl, buffer=10)[0]
        # Only keep receivers within the model'
        xmin = model.origin[0]
        idx_xrec = np.where(rec_coord[:, 0] < xmin)[0]
        is_empty = idx_xrec.size == 0
        if not is_empty:
            rec_coord = np.delete(rec_coord, idx_xrec, axis=0)

        if rec_coord.shape[0] < 2*nbl:
            s = 'Shot {0:d} located too closely to the origin, therefore,'\
                'it is not modeled'
            print('s'.format(shot_dict['id']))
            return True

        # Geometry for current shot
        geometry = AcquisitionGeometry(model, rec_coord, src_coord, t0, tn,
                                       f0=f0, src_type='Ricker')

        # Set up solver.
        if setup_func == 'tti':
            solver = AnisotropicWaveSolver(model, geometry, space_order=space_order)
        elif setup_func == 'viscoacoustic':
            solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order,
                                             time_order=1, kernel='sls')
        else:
            solver = AcousticWaveSolver(model, geometry, space_order=space_order)

        if solver_params['born']:
            gaussian_smooth(model0.vp, sigma=(5, 5))
            dm = (model.vp.data**(-2) - model0.vp.data**(-2))

            # Generate synthetic data from true model
            dobs = solver.jacobian(dm, vp=model0.vp)[0]
        else:
            autotune = ('aggressive', 'runtime') if len(shape) == 3 else False
            dobs = solver.forward(autotune=autotune)[0]

        print('Shot with time interval of {} ms'.format(model.critical_dt))

        str_shot = str(shot_dict['id']).zfill(3)
        filename = '{}_{}_suheader_{}.segy'.format('shot', str_shot, model_name)
        filename = solver_params['shotfile_path'] + filename
        if dt is not None:
            nsamples = int((tn-t0)/dt + 1)
            data = dobs.resample(num=nsamples)
        else:
            dt = model.critical_dt
            data = dobs
        # Save shot in segy format
        if len(shape) == 3:
            segy_write(data.data[:], [geometry.src.coordinates.data[0, 0]],
                       [geometry.src.coordinates.data[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename,
                       sourceY=[geometry.src.coordinates.data[0, 1]],
                       groupY=data.coordinates.data[:, -1])
        else:
            segy_write(data.data[:], [geometry.src.coordinates.data[0, 0]],
                       [geometry.src.coordinates.data[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename)

        return True

    @staticmethod
    def gen_grad_in_worker(shot_dict, solver_params):

        dtype = solver_params['dtype']

        if dtype == 'float32':
            dtype = np.float32
        elif dtype == 'float64':
            dtype = np.float64
        else:
            raise ValueError("Invalid dtype")

        origin = solver_params['origin']
        shape = solver_params['shape']
        spacing = solver_params['spacing']
        setup_func = solver_params['setup_func']

        #
        hf = h5py.File('vp_current.h5', 'r')
        vp_updt = np.zeros(shape, dtype=dtype)
        hf['current_dataset'].read_direct(vp_updt)
        hf.close()
        vp_updt = 1.0/np.sqrt(vp_updt)

        check_par_attr(DaskCluster.gen_grad_in_worker,
                       solver_params['parfile_path'], setup_func, shape)

        # Get a single shot as a numpy array
        retrieved_shot, tn, dt = DaskCluster.load_shot(shot_dict['filename'],
                                                       shot_dict['Trace_Position'],
                                                       shot_dict['Num_Traces'])

        if len(shape) == 3:
            src_coord = np.array(shot_dict['Source']).reshape((1, 3))
            rec_coord = np.array(shot_dict['Receivers'])
        else:
            src_coord = np.array([shot_dict['Source'][0],
                                  shot_dict['Source'][-1]]).reshape((1, 2))
            rec_coord = np.array([(r[0], r[-1]) for r in shot_dict['Receivers']])

        space_order = solver_params['space_order']
        nbl = solver_params['nbl']
        f0 = solver_params['f0']
        t0 = solver_params['t0']

        rfl = None
        model, _ = limit_model_to_receiver_area(rec_coord, src_coord, origin,
                                                spacing, shape, vp_updt,
                                                DaskCluster.gen_grad_in_worker.params,
                                                space_order, nbl, rfl, 10)

        # Only keep receivers within the model'
        xmin = model.origin[0]
        idx_xrec = np.where(rec_coord[:, 0] < xmin)[0]
        is_empty = idx_xrec.size == 0

        if not is_empty:
            idx_tr = np.where(rec_coord[:, 0] >= xmin)[0]
            rec_coord = np.delete(rec_coord, idx_xrec, axis=0)

        # Geometry for current shot
        geometry = AcquisitionGeometry(model, rec_coord, src_coord, 0, tn,
                                       f0=f0, src_type='Ricker')

        # Set up solver.
        if setup_func == 'tti':
            solver = AnisotropicWaveSolver(model, geometry, space_order=space_order)
        else:
            solver = AcousticWaveSolver(model, geometry, space_order=space_order)

        src_illum = Function(name='src_illum', grid=model.grid)
        grad_sgle = Function(name='grad', grid=model.grid)
        eps = np.finfo(dtype).eps

        rev_op = DaskCluster.ImagingOperator(geometry, model, grad_sgle, src_illum,
                                             space_order, setup_func)

        # For illustrative purposes, assuming that there is enough memory
        du = TimeFunction(name='du', grid=model.grid, time_order=2,
                          space_order=space_order)
        if setup_func == 'tti':
            dv = TimeFunction(name='dv', grid=model.grid, time_order=2,
                              space_order=space_order)
            rec0, u0, v0 = solver.forward(vp=model.vp, save=True)[0:-1]
        else:
            rec0, u0 = solver.forward(vp=model.vp, save=True)[0:2]

        time_range = TimeAxis(start=0, stop=tn, step=dt)
        dobs = Receiver(name='dobs', grid=model.grid, time_range=time_range,
                        coordinates=geometry.rec_positions)
        if not is_empty:
            dobs.data[:] = retrieved_shot[:, idx_tr]
        else:
            dobs.data[:] = retrieved_shot[:]
        dobs_resam = dobs.resample(num=geometry.nt)

        residual = Receiver(name='residual', grid=solver.model.grid,
                            data=rec0.data - dobs_resam.data,
                            time_range=solver.geometry.time_axis,
                            coordinates=solver.geometry.rec_positions)

        objective = .5*np.linalg.norm(residual.data.ravel())**2

        if setup_func == 'tti':
            rev_op(u0=u0, v0=v0, du=du, dv=dv, epsilon=model.epsilon,
                   delta=model.delta, theta=model.theta, vp=model.vp,
                   dt=model.critical_dt, rec=residual)
        else:
            rev_op(u0=u0, du=du, vp=model.vp, dt=model.critical_dt, rec=residual)

        eq = Eq(src_illum, grad_sgle/(src_illum+eps))
        op = Operator(eq)()
        grad = extend_image(origin, vp_updt, model, src_illum)

        del u0
        if setup_func == 'tti':
            del v0
        del solver

        return objective, grad

    @staticmethod
    def grad_lsrtm_in_worker(shot_dict, solver_params):

        dtype = solver_params['dtype']

        if dtype == 'float32':
            dtype = np.float32
        elif dtype == 'float64':
            dtype = np.float64
        else:
            raise ValueError("Invalid dtype")

        origin = solver_params['origin']
        shape = solver_params['shape']
        spacing = solver_params['spacing']
        setup_func = solver_params['setup_func']

        #
        hf = h5py.File('img_current.h5', 'r')
        X = np.zeros(shape, dtype=dtype)
        hf['current_dataset'].read_direct(X)
        hf.close()

        vp = np.empty(shape)
        pars = ['vp_smooth']
        params = [vp]
        if setup_func == 'tti':
            epsilon = np.empty(shape)
            delta = np.empty(shape)
            theta = np.empty(shape)
            pars.extend(['delta', 'epsilon', 'theta'])
            params.extend([delta, epsilon, theta])
            if len(shape) == 3:
                phi = np.empty(shape)
                pars.extend(['phi'])
                params.extend([phi])

        # Read parameters
        for file, par in zip(pars, params):
            with h5py.File(solver_params['parfile_path']+file+'.h5', 'r') as f:
                par[:] = f[file][()]

        if setup_func == 'tti':
            theta *= (np.pi/180.)  # use radians

        # Get a single shot as a numpy array
        retrieved_shot, tn, dt = DaskCluster.load_shot(shot_dict['filename'],
                                                       shot_dict['Trace_Position'],
                                                       shot_dict['Num_Traces'])

        if len(shape) == 3:
            src_coord = np.array(shot_dict['Source']).reshape((1, 3))
            rec_coord = np.array(shot_dict['Receivers'])
        else:
            src_coord = np.array([shot_dict['Source'][0],
                                  shot_dict['Source'][-1]]).reshape((1, 2))
            rec_coord = np.array([(r[0], r[-1]) for r in shot_dict['Receivers']])

        space_order = solver_params['space_order']
        nbl = solver_params['nbl']
        f0 = solver_params['f0']
        t0 = solver_params['t0']

        rfl = None
        params = params[1:] if setup_func == 'tti' else None
        model, rfl_trimmed = limit_model_to_receiver_area(rec_coord, src_coord, origin,
                                                          spacing, shape, vp, params,
                                                          space_order, nbl, rfl, 10)

        # Only keep receivers within the model'
        xmin = model.origin[0]
        idx_xrec = np.where(rec_coord[:, 0] < xmin)[0]
        is_empty = idx_xrec.size == 0

        if not is_empty:
            idx_tr = np.where(rec_coord[:, 0] >= xmin)[0]
            rec_coord = np.delete(rec_coord, idx_xrec, axis=0)

        # Geometry for current shot
        geometry = AcquisitionGeometry(model, rec_coord, src_coord, 0, tn,
                                       f0=f0, src_type='Ricker')

        # Set up solver.
        if setup_func == 'tti':
            solver = AnisotropicWaveSolver(model, geometry, space_order=space_order)
        else:
            solver = AcousticWaveSolver(model, geometry, space_order=space_order)

        grad = Function(name='grad', grid=model.grid)
        rfl_sgle = Function(name="rfl", grid=model.grid)
        rfl_sgle.data[:, :] = X
        src_illum = Function(name='src_illum', grid=model.grid)
        eps = np.finfo(dtype).eps

        op_imaging = ImagingOperator(geometry, model, grad, src_illum,
                                     space_order, setup_func)

        # For illustrative purposes, assuming that there is enough memory
        du = TimeFunction(name='du', grid=model.grid, time_order=2,
                          space_order=space_order)
        if setup_func == 'tti':
            dv = TimeFunction(name='dv', grid=model.grid, time_order=2,
                              space_order=space_order)
            rec0, u0, v0 = solver.forward(vp=model.vp, save=True)[0:-1]
        else:
            rec0, u0 = solver.forward(vp=model.vp, save=True)[0:2]

        time_range = TimeAxis(start=0, stop=tn, step=dt)
        dobs = Receiver(name='dobs', grid=model.grid, time_range=time_range,
                        coordinates=geometry.rec_positions)
        if not is_empty:
            dobs.data[:] = retrieved_shot[:, idx_tr]
        else:
            dobs.data[:] = retrieved_shot[:]
        dobs_resam = dobs.resample(num=geometry.nt)

        rec0 = solver.jacobian(rfl_sgle, vp=model.vp)[0]
        residual = Receiver(name='residual', grid=solver.model.grid,
                            data=rec0.data - dobs_resam.data,
                            time_range=solver.geometry.time_axis,
                            coordinates=solver.geometry.rec_positions)

        objective = .5*np.linalg.norm(residual.data.ravel())**2

        if setup_func == 'tti':
            op_imaging(u0=u0, v0=v0, du=du, dv=dv, epsilon=model.epsilon,
                       delta=model.delta, theta=model.theta, vp=model.vp,
                       dt=model.critical_dt, rec=residual)
        else:
            op_imaging(u0=u0, du=du, vp=model.vp, dt=model.critical_dt, rec=residual)

        eq = Eq(src_illum, grad_sgle/(src_illum+eps))
        op = Operator(eq)()
        grad = extend_image(origin, vp, model, src_illum)

        del u0
        if setup_func == 'tti':
            del v0
        del solver
        del op_imaging

        return objective, grad

    @staticmethod
    def gen_grad_in_worker_ckp(shot_dict, solver_params):

        dtype = solver_params['dtype']

        if dtype == 'float32':
            dtype = np.float32
        elif dtype == 'float64':
            dtype = np.float64
        else:
            raise ValueError("Invalid dtype")

        origin = solver_params['origin']
        shape = solver_params['shape']
        spacing = solver_params['spacing']
        setup_func = solver_params['setup_func']

        #
        hf = h5py.File('vp_current.h5', 'r')
        vp_updt = np.zeros(shape, dtype=dtype)
        hf['current_dataset'].read_direct(vp_updt)
        hf.close()
        vp_updt = 1.0/np.sqrt(vp_updt)

        check_par_attr(DaskCluster.gen_grad_in_worker_ckp,
                       solver_params['parfile_path'], setup_func, shape)

        # Get a single shot as a numpy array
        retrieved_shot, tn, dt = DaskCluster.load_shot(shot_dict['filename'],
                                                       shot_dict['Trace_Position'],
                                                       shot_dict['Num_Traces'])

        if len(shape) == 3:
            src_coord = np.array(shot_dict['Source']).reshape((1, 3))
            rec_coord = np.array(shot_dict['Receivers'])
        else:
            src_coord = np.array([shot_dict['Source'][0],
                                  shot_dict['Source'][-1]]).reshape((1, 2))
            rec_coord = np.array([(r[0], r[-1]) for r in shot_dict['Receivers']])

        space_order = solver_params['space_order']
        nbl = solver_params['nbl']
        f0 = solver_params['f0']
        t0 = solver_params['t0']

        rfl = None
        model, _ = limit_model_to_receiver_area(rec_coord, src_coord, origin,
                                                spacing, shape, vp_updt,
                                                DaskCluster.gen_grad_in_worker_ckp.params,
                                                space_order, nbl, rfl, 10)
        # gaussian_smooth(model.vp, sigma=(5, 5))
        # Only keep receivers within the model'
        xmin = model.origin[0]
        idx_xrec = np.where(rec_coord[:, 0] < xmin)[0]
        is_empty = idx_xrec.size == 0

        if not is_empty:
            idx_tr = np.where(rec_coord[:, 0] >= xmin)[0]
            rec_coord = np.delete(rec_coord, idx_xrec, axis=0)

        # Geometry for current shot
        geometry = AcquisitionGeometry(model, rec_coord, src_coord, 0, tn,
                                       f0=f0, src_type='Ricker')

        # Set up solver.
        save = False
        solver = AcousticWaveSolver(model, geometry, space_order=space_order)
        du = TimeFunction(name='du', grid=model.grid, time_order=2,
                          space_order=space_order)
        l = [du]

        grad_sgle = Function(name='grad', grid=model.grid)
        src_illum = Function(name='src_illum', grid=model.grid)
        eps = np.finfo(dtype).eps

        rev_op = DaskCluster.ImagingOperator(geometry, model, grad_sgle, src_illum,
                                             space_order, setup_func, save=save)

        if setup_func == 'tti':
            solver = AnisotropicWaveSolver(model, geometry, space_order=space_order)
            dv = TimeFunction(name='dv', grid=model.grid, time_order=2,
                              space_order=space_order)
            l.extend([dv])

        time_range = TimeAxis(start=0, stop=tn, step=dt)
        dobs = Receiver(name='dobs', grid=model.grid, time_range=time_range,
                        coordinates=geometry.rec_positions)
        rec0 = Receiver(name='rec0', grid=model.grid, time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
        residual = Receiver(name='residual', grid=model.grid,
                            time_range=geometry.time_axis,
                            coordinates=geometry.rec_positions)
        if not is_empty:
            dobs.data[:] = retrieved_shot[:, idx_tr]
        else:
            dobs.data[:] = retrieved_shot[:]
        dobs_resam = dobs.resample(num=geometry.nt)

        cp = DevitoCheckpoint(l)
        fwd_op = solver.op_fwd(save=save)
        fwd_op.cfunction
        rev_op.cfunction

        if setup_func == 'tti':
            wrap_fw = CheckpointOperator(fwd_op, src=geometry.src, rec=rec0,
                                         u=du, v=dv, vp=model.vp, epsilon=model.epsilon,
                                         delta=model.delta, theta=model.theta,
                                         dt=model.critical_dt)
            wrap_rev = CheckpointOperator(rev_op, u0=du, v0=dv, vp=model.vp,
                                          epsilon=model.epsilon, delta=model.delta,
                                          theta=model.theta, dt=model.critical_dt,
                                          rec=residual)
        else:
            wrap_fw = CheckpointOperator(fwd_op, src=geometry.src, rec=rec0,
                                         u=du, vp=model.vp, dt=model.critical_dt)
            wrap_rev = CheckpointOperator(rev_op, u0=du, vp=model.vp,
                                          dt=model.critical_dt, rec=residual)

        # Set Revolve
        if solver_params['pct_ckp'] is not None:
            pct_ckp = solver_params['pct_ckp']
        else:
            pct_ckp = 0.01
        n_checkpoints = int(pct_ckp * geometry.nt)

        # Run forward
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, dobs_resam.shape[0]-2)
        wrp.apply_forward()

        # Compute gradient from data residual and update objective function
        residual.data[:] = rec0.data[:] - dobs_resam.data[:]

        objective = .5*np.linalg.norm(residual.data.ravel())**2
        wrp.apply_reverse()

        eq = Eq(src_illum, grad_sgle/(src_illum+eps))
        op = Operator(eq)()
        grad = extend_image(origin, vp_updt, model, src_illum)

        del op
        del solver
        del rev_op
        del wrp
        gc.collect()

        return objective, grad

    @staticmethod
    def load_shot(filename, position, traces_in_shot):

        f = segyio.open(filename, ignore_geometry=True)
        num_samples = len(f.samples)
        samp_int = f.bin[segyio.BinField.Interval]/1000.
        retrieved_shot = np.zeros((num_samples, traces_in_shot))
        shot_traces = f.trace[position:position+traces_in_shot]
        for i, trace in enumerate(shot_traces):
            retrieved_shot[:, i] = trace

        tmax = (num_samples-1)*samp_int

        return retrieved_shot, tmax, samp_int

    @staticmethod
    def ImagingOperator(geometry, model, image, src_illum, space_order,
                        setup_func, save=True):

        dt = geometry.dt
        time_order = 2

        rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nrec)

        # Gradient symbol and wavefield symbols
        u0 = TimeFunction(name='u0', grid=model.grid, save=geometry.nt if save
                          else None, time_order=time_order, space_order=space_order)
        du = TimeFunction(name="du", grid=model.grid, save=None,
                          time_order=time_order, space_order=space_order)
        if setup_func == 'tti':
            v0 = TimeFunction(name='v0', grid=model.grid, save=geometry.nt if save
                              else None, time_order=time_order, space_order=space_order)
            dv = TimeFunction(name="dv", grid=model.grid, save=None,
                              time_order=time_order, space_order=space_order)
            # FD kernels of the PDE
            eqn = kernel_centered_2d(model, du, dv, space_order, forward=False)

            image_update = Inc(image, - (u0.dt2 * du + v0.dt2 * dv))
            src_illum_updt = Eq(src_illum, src_illum + (u0 + v0)**2)

            # Add expression for receiver injection
            rec_term = rec.inject(field=du.backward, expr=rec * dt**2 / model.m)
            rec_term += rec.inject(field=dv.backward, expr=rec * dt**2 / model.m)

            # Substitute spacing terms to reduce flops
            return Operator(eqn + rec_term + [image_update] +
                            [src_illum_updt], subs=model.spacing_map)
        else:
            # Define the wave equation, but with a negated damping term
            eqn = iso_stencil(du, model, kernel='OT2', forward=False)

            # Define residual injection at the location of the forward receivers
            res_term = rec.inject(field=du.backward, expr=rec * dt**2 / model.m)

            # Correlate u and v for the current time step and add it to the image
            image_update = Inc(image, - u0.dt2 * du)
            src_illum_updt = Eq(src_illum, src_illum + u0**2)

            return Operator(eqn + res_term + [image_update] +
                            [src_illum_updt], subs=model.spacing_map)

