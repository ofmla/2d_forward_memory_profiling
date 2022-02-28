#!/usr/bin/env python
# coding: utf-8

# basic imports.
import numpy as np
import segyio as so


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
