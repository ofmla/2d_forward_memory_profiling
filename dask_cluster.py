#!/usr/bin/env python
# coding: utf-8

# basic imports.
import os
import errno
import numpy as np
import time
import yaml
import json
import h5py
import gc

from dask_jobqueue import SLURMCluster
# import multiprocessing.popen_spawn_posix
from dask.distributed import Client, LocalCluster

from devito import *

from examples.seismic import AcquisitionGeometry, SeismicModel
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import TimeAxis, RickerSource, Receiver

from utils import segy_write, humanbytes


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
            self.config_values["memory"] = 2.5
        if "job_extra" not in self.config_values:
            self.config_values["job_extra"] = ['-e slurm-%j.err', '-o slurm-%j.out',
                                               '--job-name="dask_task"']

        if self.config_values["use_local_cluster"]:
            # single-threaded execution, as this is actually best for the workload
            cluster = LocalCluster(n_workers=self.config_values["n_workers"],
                                   threads_per_worker=1,
                                   memory_limit=str(self.config_values["memory"])+"GB",
                                   death_timeout=60,
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

        # Wait for cluster to start
        time.sleep(30)
        self.client = Client(cluster)
        # initialize tasks dictionary
        self._set_tasks_from_files()

    def _set_tasks_from_files(self):
        "Returns a dict which contains the list of tasks to be run"

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

    def launch_tasks(self, func):

        shot_futures = []
        all_shot_results = []
        par = self.client.scatter({**self.config_values['solver_params']}, broadcast=True)

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
        model_name = solver_params['model_name']

        #
        vp = np.empty(shape)
        # Read medium parameters
        with h5py.File(solver_params['parfile_path']+'vp.h5', 'r') as f:
            vp[:] = f['vp'][()]

        # Get src/rec coords
        src_coord = np.array(shot_dict['Source']).reshape((1, len(shape)))
        rec_coord = np.array(shot_dict['Receivers'])

        space_order = solver_params['space_order']
        dt = solver_params['dt']
        nbl = solver_params['nbl']
        f0 = solver_params['f0']
        t0 = solver_params['t0']
        tn = solver_params['tn']
        use_solver = solver_params['use_solver']

        model = SeismicModel(vp=vp, origin=origin, shape=vp.shape, spacing=spacing,
                             space_order=space_order, nbl=nbl, bcs="damp", dtype=dtype)

        autotune = ('aggressive', 'runtime') if len(shape) == 3 else False
        if use_solver:
            # Geometry for current shot
            geometry = AcquisitionGeometry(model, rec_coord, src_coord, t0, tn,
                                           f0=f0, src_type='Ricker')
            # Set up solver.
            solver = AcousticWaveSolver(model, geometry, space_order=space_order)
            dobs = solver.forward(autotune=autotune)[0]
        else:
            # Create the forward operator step by step as in tutorial 01
            dt2 = model.critical_dt
            time_range = TimeAxis(start=t0, stop=tn, step=dt2)
            src = RickerSource(name='src', grid=model.grid, f0=f0,
                               npoint=1, time_range=time_range)
            src.coordinates.data[:] = src_coord
            dobs = Receiver(name='dobs', grid=model.grid, npoint=rec_coord.shape[0],
                            time_range=time_range)
            dobs.coordinates.data[:] = rec_coord

            # Define the wavefield with the size of the model and the time dimension
            u = TimeFunction(name="u", grid=model.grid, time_order=2,
                             space_order=space_order)
            pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
            stencil = Eq(u.forward, solve(pde, u.forward))
            #
            src_term = src.inject(field=u.forward, expr=src * dt2**2 / model.m)
            rec_term = dobs.interpolate(expr=u.forward)
            op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

            op(time=time_range.num-1, dt=model.critical_dt, autotune=autotune)

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
            segy_write(data.data[:], [src_coord[0, 0]],
                       [src_coord[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename,
                       sourceY=[src_coord[0, 1]],
                       groupY=data.coordinates.data[:, -1])
        else:
            segy_write(data.data[:], [src_coord[0, 0]],
                       [src_coord[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename)
        # del u
        # clear_cache()
        # gc.collect()

        return True
