#!/usr/bin/env python
# coding: utf-8

# basic imports.
import numpy as np
import math
import segyio as so
import csv
import socket
import os
import h5py
import json

from examples.seismic import SeismicModel


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def extend_image(origin, vp, model, image):
    "Extend image back to full model size"
    ndim = len(origin)
    full_image = np.zeros(vp.shape)
    nx_start = math.trunc(((model.origin[0] - origin[0])/model.spacing[0]))
    nx_end = nx_start + model.vp.shape[0]-2*model.nbl
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(ndim))
    if ndim == 2:
        full_image[nx_start:nx_end, :] = image.data[slices]
    else:
        ny_start = math.trunc(((model.origin[1] - origin[1])/model.spacing[1]) + 1)
        ny_end = ny_start + model.vp.shape[1]-2*model.nbl
        full_image[nx_start:nx_end, ny_start:ny_end, :] = image.data[slices]

    return full_image


def segy_write(data, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None,
               groupY=None, elevScalar=-1000, coordScalar=-1000):

    nt = data.shape[0]
    nxrec = len(groupX)

    if sourceY is None and groupY is None:
        sourceY = np.zeros(1, dtype='int')
        groupY = np.zeros(nxrec, dtype='int')

    # Create spec object
    spec = so.spec()
    spec.ilines = np.arange(nxrec)    # dummy trace count
    spec.xlines = np.zeros(1, dtype='int')  # assume coords are already vectorized for 3D
    spec.samples = range(nt)
    spec.format = 1
    spec.sorting = 1
    with so.create(filename, spec) as segyfile:
        for i in range(nxrec):
            segyfile.bin = {
                so.BinField.Samples: data.shape[0],
                so.BinField.Traces: data.shape[1],
                so.BinField.Interval: int(dt*1e3)
            }
            segyfile.header[i] = {
                so.su.tracl: i+1,
                so.su.tracr: i+1,
                so.su.fldr: 1,
                so.su.tracf: i+1,
                so.su.sx: int(np.round(sourceX[0] * np.abs(coordScalar))),
                so.su.sy: int(np.round(sourceY[0] * np.abs(coordScalar))),
                so.su.selev: int(np.round(sourceZ[0] * np.abs(elevScalar))),
                so.su.gx: int(np.round(groupX[i] * np.abs(coordScalar))),
                so.su.gy: int(np.round(groupY[i] * np.abs(coordScalar))),
                so.su.gelev: int(np.round(groupZ[i] * np.abs(elevScalar))),
                so.su.dt: int(dt*1e3),
                so.su.scalel: int(elevScalar),
                so.su.scalco: int(coordScalar)
            }
            segyfile.trace[i] = data[:, i]
        segyfile.dt = int(dt*1e3)


def segy_read(filename, ndims=2):

    with so.open(filename, "r", ignore_geometry=True) as segyfile:
        segyfile.mmap()

        # Assume input data is for single common shot gather
        sourceX = segyfile.attributes(so.TraceField.SourceX)[0]
        sourceY = segyfile.attributes(so.TraceField.SourceY)[0]
        sourceZ = segyfile.attributes(so.TraceField.SourceSurfaceElevation)[0]
        groupX = segyfile.attributes(so.TraceField.GroupX)[:]
        groupY = segyfile.attributes(so.TraceField.GroupY)[:]
        groupZ = segyfile.attributes(so.TraceField.ReceiverGroupElevation)[:]
        dt = so.dt(segyfile)/1e3

        # Apply scaling
        elevSc = segyfile.attributes(so.TraceField.ElevationScalar)[0]
        coordSc = segyfile.attributes(so.TraceField.SourceGroupScalar)[0]

        if coordSc < 0.:
            sourceX = sourceX / np.abs(coordSc)
            sourceY = sourceY / np.abs(coordSc)
            sourceZ = sourceZ / np.abs(elevSc)
            groupX = groupX / np.abs(coordSc)
            groupY = groupY / np.abs(coordSc)
        elif coordSc > 0.:
            sourceX = sourceX * np.abs(coordSc)
            sourceY = sourceY * np.abs(coordSc)
            sourceZ = sourceZ * np.abs(elevSc)
            groupX = groupX * np.abs(coordSc)
            groupY = groupY / np.abs(coordSc)

        if elevSc < 0.:
            groupZ = groupZ / np.abs(elevSc)
        elif elevSc > 0.:
            groupZ = groupZ * np.abs(elevSc)

        nrec = len(groupX)
        nt = len(segyfile.trace[0])

        # Extract data
        data = np.zeros(shape=(nt, nrec), dtype='float32')
        for i in range(nrec):
            data[:, i] = segyfile.trace[i]
        tmax = (nt-1)*dt

    if ndims == 2:
        return data, np.vstack((sourceX, sourceZ)).T,
        np.vstack((groupX, groupZ)).T, tmax, dt, nt
    else:
        return data, np.vstack((sourceX, sourceY, sourceZ)).T,
        np.vstack((groupX, groupY, groupZ)).T, tmax, dt, nt


def make_lookup_table(sgy_file):
    '''
    Make a lookup of shots, where the keys are the shot record IDs being
    searched (looked up)
    '''
    tbl = {}
    with so.open(sgy_file, ignore_geometry=True) as f:
        f.mmap()
        idx = None
        pos_in_file = 0
        for hdr in f.header:
            if int(hdr[so.TraceField.SourceGroupScalar]) < 0:
                scalco = abs(1./hdr[so.TraceField.SourceGroupScalar])
            else:
                scalco = hdr[so.TraceField.SourceGroupScalar]
            if int(hdr[so.TraceField.ElevationScalar]) < 0:
                scalel = abs(1./hdr[so.TraceField.ElevationScalar])
            else:
                scalel = hdr[so.TraceField.ElevationScalar]
            # Check to see if we're in a new shot
            if idx != hdr[so.TraceField.FieldRecord]:
                idx = hdr[so.TraceField.FieldRecord]
                tbl[idx] = {}
                tbl[idx]['filename'] = sgy_file
                tbl[idx]['Trace_Position'] = pos_in_file
                tbl[idx]['Num_Traces'] = 1
                tbl[idx]['Source'] = (hdr[so.TraceField.SourceX]*scalco,
                                      hdr[so.TraceField.SourceY]*scalco,
                                      hdr[so.TraceField.SourceSurfaceElevation] *
                                      scalel)
                tbl[idx]['Receivers'] = []
            else:  # Not in a new shot, so increase the number of traces in the shot by 1
                tbl[idx]['Num_Traces'] += 1
            tbl[idx]['Receivers'].append((hdr[so.TraceField.GroupX]*scalco,
                                         hdr[so.TraceField.GroupY]*scalco,
                                         hdr[so.TraceField.ReceiverGroupElevation] *
                                         scalel))
            pos_in_file += 1

    return tbl


def create_shot_dict(table, origin, extent):
    shot_dict = {}
    for key, value in table.items():
        if value['Source'][0] > origin and value['Source'][0] < extent:
            shot_dict[key] = table[key]

    return shot_dict


def save_model(model_name, datakey, data, metadata, dtype=np.float32):
    with h5py.File(model_name, 'w') as f:
        f.create_dataset(datakey, data=data, dtype=dtype)
        f.create_dataset('metadata', data=json.dumps(metadata))


def load_shot(filename, position, traces_in_shot):
    f = so.open(filename, ignore_geometry=True)
    num_samples = len(f.samples)
    samp_int = f.bin[so.BinField.Interval]/1000.
    retrieved_shot = np.zeros((num_samples, traces_in_shot))
    shot_traces = f.trace[position:position+traces_in_shot]
    for i, trace in enumerate(shot_traces):
        retrieved_shot[:, i] = trace

    tmax = (num_samples-1)*samp_int

    return retrieved_shot, tmax, samp_int


def limit_model_to_receiver_area(rec_coord, src_coord, origin, spacing, shape,
                                 vel, par=None, space_order=8, nbl=40, rfl=None,
                                 buffer=0):
    '''
    Restrict full velocity model to area that contains either sources and
    receivers
    '''
    ndim = len(origin)
    rfl_trimmed = None

    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = min(src_coord[0][0], np.amin(rec_coord[:, 0]))
    max_x = max(src_coord[0][0], np.amax(rec_coord[:, 0]))
    # print(min_x,max_x)
    if ndim == 3:
        min_y = min(src_coord[0][1], np.amin(rec_coord[:, 1]))
        max_y = max(src_coord[0][1], np.amax(rec_coord[:, 0]))

    # add buffer zone if possible
    min_x = max(origin[0], min_x-buffer)
    max_x = min(origin[0] + spacing[0]*(shape[0]-1), max_x+buffer)
    # print(min_x,max_x)
    if ndim == 3:
        min_y = max(origin[1], min_y-buffer)
        max_y = min(origin[1] + spacing[1]*(shape[1]-1), max_y+buffer)

    # extract part of the model that contains sources/receivers
    nx_min = int(round((min_x - origin[0])/spacing[0]))
    nx_max = int(round((max_x - origin[0])/spacing[0])+1)
    # print(nx_min,nx_max)

    if ndim == 2:
        ox = origin[0]+float((nx_min)*spacing[0])
        oz = origin[-1]
    else:
        ny_min = round(min_y/spacing[1])
        ny_max = round(max_y/spacing[1])+1
        ox = float((nx_min)*spacing[0])
        oy = float((ny_min)*spacing[1])
        oz = origin[-1]

    # Extract relevant model part from full domain
    delta = epsilon = theta = phi = None
    if ndim == 2:
        if rfl is not None:
            rfl_trimmed = rfl[nx_min: nx_max, :]
        vel = vel[nx_min: nx_max, :]
        if par is not None:
            delta = par[0][nx_min: nx_max, :]
            epsilon = par[1][nx_min: nx_max, :]
            theta = par[2][nx_min: nx_max, :]
        origin = (ox, oz)
    else:
        if rfl is not None:
            rfl_trimmed = rfl[nx_min: nx_max, ny_min:ny_max, :]
        vel = vel[nx_min:nx_max, ny_min:ny_max, :]
        if par is not None:
            delta = par[0][nx_min:nx_max, ny_min:ny_max, :]
            epsilon = par[1][nx_min:nx_max, ny_min:ny_max, :]
            theta = par[2][nx_min:nx_max, ny_min:ny_max, :]
            phi = par[3][nx_min:nx_max, ny_min:ny_max, :]
        origin = (ox, oy, oz)

    model = SeismicModel(vp=vel, origin=origin, shape=vel.shape, spacing=spacing,
                         space_order=space_order, nbl=nbl, epsilon=epsilon,
                         delta=delta, theta=theta, phi=phi, bcs="damp",
                         dtype=np.float32)

    return model, rfl_trimmed


def save_timings(shape, ckp_size, nckp, tn, nt, cube_size, dtype, ckp_type,
                 storage_list, setup_func, csv_row, record):

    # log file
    logname = './log/logfile_{}.txt'.format(str(record).zfill(5))
    logfile = open(logname, "w")
    results_file = './timmings/timmings_{}.txt'.format(str(record).zfill(5))

    hostname = socket.gethostname()
    # Check whether the specified path is an existing file
    if not os.path.isfile(results_file):
        write_header = True
    else:
        write_header = False

    mckpsize = 0  # mem
    dckpsize = 0  # disk
    itemsize = np.dtype(dtype).itemsize
    full_fld_mem = cube_size * itemsize * nt
    if setup_func == 'tti':
        full_fld_mem *= 2.0

    if ckp_type == 'MULTILEVEL_CKP':
        # multilevel ckp has static size
        mckpsize = storage_list[0].maxsize  # mem
        dckpsize = storage_list[1].maxsize  # disk
    elif ckp_type == 'MEM_CKP':
        mckpsize = ckp_size*nckp
    else:
        dckpsize = ckp_size*nckp

    # 2 fields x 2 time steps for checkpoint
    ckp_fld_mem = (mckpsize + dckpsize)
    logfile.write("####################################################\n")
    logfile.write("NT: {} ms NCKP:{} \n".format(tn, nckp))
    logfile.write(
        "Mem full fld: {} bytes ({}) \n".format(
            full_fld_mem, humanbytes(full_fld_mem)
        )
    )
    logfile.write(
        "Mem ckp fld: {} bytes ({}) \n".format(ckp_fld_mem, humanbytes(ckp_fld_mem))
    )
    logfile.write(
        "Number of checkpoints/timesteps: {}/{}\n".format(
            nckp, nt
        )
    )
    logfile.write(
        "Memory saving: {}\n".format(humanbytes(full_fld_mem - ckp_fld_mem))
    )
    logfile.write(
        "Revolver storage: {}\n".format(
            humanbytes(ckp_size * nckp * itemsize)
        )
    )
    fieldnames = [
        "hostname",
        "revolver_type",
        "grid",
        "ntimesteps",
        "nckp",
        "one_ckp_size",
        "full_fld_storage",
        "ckp_storage",
        "mckp_storage",
        "dckp_storage",
        "diff_storage"
    ] + list(csv_row.keys())
    if len(shape) == 3:
        grid = '({};{};{})'.format(shape[0], shape[1], shape[2])
    else:
        grid = '({};{})'.format(shape[0], shape[-1])
    csv_row["hostname"] = hostname
    csv_row["grid"] = grid
    csv_row["ntimesteps"] = nt
    csv_row["nckp"] = nckp
    csv_row["revolver_type"] = ckp_type
    csv_row["one_ckp_size"] = ckp_size
    csv_row["full_fld_storage"] = full_fld_mem
    csv_row["ckp_storage"] = ckp_fld_mem
    csv_row["mckp_storage"] = mckpsize
    csv_row["dckp_storage"] = dckpsize
    csv_row["diff_storage"] = full_fld_mem-ckp_fld_mem

    with open(results_file, "a") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)


def check_par_attr(someobject, filepath, setup_func, shape, fwi=True):

    property = getattr(someobject, 'params', None)
    if property is None:
        if setup_func == 'tti':
            if not fwi:
                vp = np.empty(shape)
                pars = ['approx_vp']
                someobject.params = [vp]
            else:
                someobject.params = []
                pars = []
            epsilon = np.empty(shape)
            delta = np.empty(shape)
            theta = np.empty(shape)
            pars.extend(['delta', 'epsilon', 'theta'])
            someobject.params.extend([delta, epsilon, theta])
            if len(shape) == 3:
                phi = np.empty(shape)
                pars.extend(['phi'])
                someobject.params.extend([phi])
        else:
            if not fwi:
                vp = np.empty(shape)
                pars = ['approx_vp']
                someobject.params = [vp]
        # Read parameters
        for file, par in zip(pars, someobject.params):
            with h5py.File(filepath+file+'.h5', 'r') as f:
                par[:] = f[file][()]
                
        if setup_func == 'tti':
        	theta *= (np.pi/180.)  # use radians	
