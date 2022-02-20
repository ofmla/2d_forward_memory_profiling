"""
This module produces a velocity model or seismic image depending whether the full-waveform
inversion (FWI) method or the least-squares reverse time migration (LSRTM) method is used
"""
import numpy as np
import json
import h5py
from dask_cluster import DaskCluster
from sotb_wrapper import interface
from ctypes import c_int, c_float, c_bool


class ControlInversion:
    "Class to control the gradient-based inversion using sotb-wrapper"

    def run_inversion(self):
        "Run the inversion workflow"

        dask_cluster = DaskCluster()

        lb = None  # lower bound constraint
        ub = None  # upper bound constraint

        parfile_path = dask_cluster.config_values['solver_params']['parfile_path']
        s = 'vp_start' if dask_cluster.config_values['fwi'] else 'vp'

        # Read initial guess and metadata from hdf5 file
        with h5py.File(parfile_path + s + '.h5', 'r') as f:
            v0 = f[s][()]
            metadata = json.loads(f['metadata'][()])

        dask_cluster.config_values['solver_params']['origin'] = metadata['origin']
        dask_cluster.config_values['solver_params']['spacing'] = metadata['spacing']
        shape = dask_cluster.config_values['solver_params']['shape'] = metadata['shape']

        if dask_cluster.config_values['fwi']:
            X = 1.0 / (v0.reshape(-1).astype(np.float32))**2
            # dictionary_items = dask_cluster.config_values.items()
            # for item in dictionary_items:
            #    print(item)

            # Define physical constraints on velocity - we know the
            # maximum and minimum velocities we are expecting
            vmax = dask_cluster.config_values['vmax']
            vmin = dask_cluster.config_values['vmin']
            lb = np.ones((np.prod(shape),), dtype=np.float32)*1.0/vmax**2  # in [s^2/km^2]
            ub = np.ones((np.prod(shape),), dtype=np.float32)*1.0/vmin**2  # in [s^2/km^2]
        else:
            r0 = np.zeros_like(v0, dtype=np.float32)
            X = r0.reshape(-1).astype(np.float32)

        g = open('gradient_zero.file', 'wb')

        # Create an instance of the SEISCOPE optimization toolbox (sotb) Class.
        sotb = interface.sotb_wrapper()

        # Set some fields of the UserDefined derived type in Fortran (ctype structure).
        # parameter initialization
        n = c_int(np.prod(shape))       # dimension
        flag = c_int(0)                 # first flag
        sotb.udf.conv = c_float(1e-8)   # tolerance for the stopping criterion
        sotb.udf.print_flag = c_int(1)  # print info in output files
        sotb.udf.debug = c_bool(False)  # level of details for output files
        sotb.udf.niter_max = c_int(30)  # maximum iteration number
        sotb.udf.l = c_int(5)

        opt_meth = dask_cluster.config_values['opt_meth']

        # computation of the cost and gradient associated
        # with the initial guess
        fcost, grad = dask_cluster.generate_grad_in_cluster(X)
        grad_preco = np.copy(grad)

        # Save first gradient/image
        grad.reshape(-1, shape[1]).astype('float32').tofile(g)

        # Optimization loop
        while (flag.value != 2 and flag.value != 4):
            if opt_meth == 'PSTD':
                sotb.PSTD(n, X, fcost, grad, grad_preco, flag, lb, ub)
            elif opt_meth == 'PNLCG':
                sotb.PNLCG(n, X, fcost, grad, grad_preco, flag, lb, ub)
            else:
                sotb.LBFGS(n, X, fcost, grad, flag, lb, ub)
            if (flag.value == 1):
                # compute cost and gradient at point x
                fcost, grad = dask_cluster.generate_grad_in_cluster(X)
                # no preconditioning in this test: simply copy grad in
                # grad_preco
                if opt_meth != 'LBFGS':
                    grad_preco = np.copy(grad)

        # Helpful console writings
        s1 = 'vp' if dask_cluster.config_values['fwi'] else 'rfl'
        print('END OF TEST')
        print('FINAL iterate is : ', X)
        if opt_meth == 'LBFGS':
            print('See the convergence history in iterate_'+opt_meth[:2]+'.dat')
            s = s1+'_final_result_'+opt_meth[:2]
        elif opt_meth == 'PNLCG':
            print('See the convergence history in iterate_'+opt_meth[3:]+'.dat')
            s = s1+'_final_result_'+opt_meth[3:]
        else:
            print('See the convergence history in iterate_'+opt_meth[1:-1]+'.dat')
            s = s1+'_final_result_'+opt_meth[1:-1]

        # Save final model/image
        if dask_cluster.config_values['fwi']:
            X = 1./np.sqrt(X)
        g = open(s+'.file', 'wb')
        X.reshape(-1, shape[1]).astype('float32').tofile(g)
        with h5py.File(s+'.h5', 'w') as f:
            f.create_dataset('dataset', data=X.reshape(-1, shape[1]).astype('float32'))
            f.create_dataset('metadata', data=json.dumps(metadata))
