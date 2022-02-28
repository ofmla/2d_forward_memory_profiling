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

from devito import *

from examples.seismic import AcquisitionGeometry, SeismicModel
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import TimeAxis, RickerSource, Receiver

from utils import segy_write, humanbytes


def main():
    config_file = os.path.join(os.getcwd(), "config", "config.yaml")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    with open(config_file) as file:
        config_values = yaml.load(file, Loader=yaml.FullLoader)

    solver_params = config_values['solver_params']
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

    space_order = solver_params['space_order']
    dt = solver_params['dt']
    nbl = solver_params['nbl']
    f0 = solver_params['f0']
    t0 = solver_params['t0']
    tn = solver_params['tn']
    use_solver = solver_params['use_solver']

    nshots = config_values['nshots']
    nrecs = config_values['nrecs']
    model_size = config_values['model_size']
    src_step = config_values['src_step']
    # Define acquisition geometry: receivers
    # First, sources position
    src_coord = np.empty((nshots, 2))
    src_coord[:, 0] = np.arange(start=0., stop=model_size, step=src_step)
    src_coord[:, -1] = config_values['src_depth']
    # Initialize receivers for synthetic and imaging data
    rec_coord = np.empty((nrecs, 2))
    rec_coord[:, 0] = np.linspace(0, model_size, num=nrecs)
    rec_coord[:, 1] = config_values['rec_depth']

    model = SeismicModel(vp=vp, origin=origin, shape=vp.shape, spacing=spacing,
                         space_order=space_order, nbl=nbl, bcs="damp", dtype=dtype)

    autotune = ('aggressive', 'runtime') if len(shape) == 3 else False
    if use_solver:
        # Geometry for current shot
        geom = AcquisitionGeometry(model, rec_coord, np.empty((1, len(shape))), t0, tn,
                                   f0=f0, src_type='Ricker')
        # Set up solver.
        solver = AcousticWaveSolver(model, geom, space_order=space_order)
        # op = solver.op_fwd(save=None)
        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=model.grid, time_order=2,
                         space_order=space_order)
    else:
        # Create the forward operator step by step as in tutorial 01
        dt2 = model.critical_dt
        time_range = TimeAxis(start=t0, stop=tn, step=dt2)
        src = RickerSource(name='src', grid=model.grid, f0=f0,
                           npoint=1, time_range=time_range)
        dobs = Receiver(name='dobs', grid=model.grid, npoint=369,
                        time_range=time_range)

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=model.grid, time_order=2,
                         space_order=space_order)
        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
        stencil = Eq(u.forward, solve(pde, u.forward))
        #
        src_term = src.inject(field=u.forward, expr=src * dt2**2 / model.m)
        rec_term = dobs.interpolate(expr=u.forward)
        op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

    itemsize = np.dtype(dtype).itemsize
    print("Mem full fld: {} bytes ({}) \n".format(
        u.data.size * itemsize, humanbytes(u.data.size * itemsize)))

    for i in range(nshots):
        if use_solver:
            u.data[:] = 0.
            geom.src_positions[:] = src_coord[i, :]
            src_xyz = geom.src_positions[:]
            geom.rec_positions[:] = rec_coord[:]
            # dobs = geom.rec
            dobs = solver.forward(u=u, autotune=autotune)[0]
            # op =solver.op_fwd(save=None)
            # print(vars(op))
            # op(src=geom.src, rec=dobs, u=u, dt=model.critical_dt, autotune=autotune)
            # solver.op_fwd(save=None).apply(src=geom.src, rec=dobs, u=u,
            #                                 dt=model.critical_dt, autotune=autotune)
        else:
            u.data[:] = 0.
            src.coordinates.data[:] = src_coord[i, :]
            src_xyz = src.coordinates.data[:]
            dobs.coordinates.data[:] = rec_coord[:]
            op(time=time_range.num-1, dt=model.critical_dt, autotune=autotune)

        print('Shot with time interval of {} ms'.format(model.critical_dt))

        str_shot = str(i).zfill(3)
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
            segy_write(data.data[:], [src_xyz[0, 0]],
                       [src_xyz[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename,
                       sourceY=[src_coord[0, 1]],
                       groupY=data.coordinates.data[:, -1])
        else:
            segy_write(data.data[:], [src_xyz[0, 0]],
                       [src_xyz[0, -1]],
                       data.coordinates.data[:, 0],
                       data.coordinates.data[:, -1], dt, filename)

    return True


if __name__ == "__main__":
    main()
