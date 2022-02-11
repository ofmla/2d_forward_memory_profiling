# fwi-lsrtm-python

This repository provides simple Python implementations of least-squares reverse time migration (LSRTM) and full-waveform inversion (FWI) for TTI- and (visco)acoustic
media. The implementations of both inversion workflows are based on the [Devito](https://www.devitoproject.org/), [sotb-wrapper](https://github.com/ofmla/seiscope_opt_toolbox_w_ctypes) and [Dask](https://dask.org/) libraries. The [h5py](https://github.com/h5py/h5py), [segyio](https://github.com/equinor/segyio) and [pyrevolve](https://github.com/devitocodes/pyrevolve) packages are also used. The 2-D TTI Marmousi model, 2-D TTI Overthrust model and 
the 2-D TTI Marmousi II model are used to prove the robustness of the implementations.

## Forward modeling

For generating synthetic seismic data, you are first required to edit the YAML file (`config.yaml`) in the `config` folder to set the `forward` key to `true`, change parameters for configuring the geometry and set the correct absolute paths in the variables `parfile_path` and `shotfile_path` of dictionary `solver_params`. They point to the folders were the physical model parameters (in [hdf5](https://www.hdfgroup.org/solutions/hdf5) format) are and shot files (in [segy](https://wiki.seg.org/wiki/SEG-Y) format) will be saved, respectively. Further control parameters need to be specified in the dictionary, including the minimum and maximum recording times `t0` and `tn`, sampling rate `dt`, peak frequency `f0`, order of accuracy of the finite difference scheme for spatial derivatives `space_order`, number of layers of the absorbing boundary condition `nbl` and floating-point precision to be used by Devito in code generation. In these settings you can choose from three modeling modes (`acoustic`, `viscoacoustic` and `tti`) to produce the seismic data. Set the `setup_func` to your preference. Born modeling can also be enabled by setting the `born` key to `true`. 

See below definitions in `config.yaml` to create synthetic data using the Marmousi II model. Note that both, sources and receivers are evenly distributed over the x-dimension. This means that you only need to specify the number for each of them `nshots` and `nrecs`, and their depths `src_depth` and `rec_depth`. 
``` yaml
# geometry
file_src: null
file_rec: null
nshots: 16 
src_depth: 40.
src_step: 1133.
nrecs: 426
rec_depth: 80.
model_size: 17000.0

# solver parameters
solver_params:
  shotfile_path: "./"
  parfile_path: "./marmousi2/parameters_hdf5/"
  setup_func: "tti"
  t0: 0.0
  tn: 5000.0 
  dt: 4.0
  f0: 0.004 
  model_name: "marmousi2" 
  nbl: 50
  space_order: 8
  born: false
  dtype: "float32"
```
Once your settings are complete, you must set up the dask cluster in `config.yaml`. Depending on available hardware, you can launch a local Dask cluster (on your laptop or single computing node) or a non-local one (on a distributed HPC with job scheduler installed). In latter case we only support an HPC cluster which uses the SLURM scheduler. Set `use_local_cluster` to `false` or `true`, depending of the hardware. Here are the settings for running the forward modeling of 16 shots for the Marmousi II model, on a quad core laptop (i.e., Intel(R) Core(TM) i7-8565U CPU @1.80GHz)

``` yaml
# dask cluster
use_local_cluster: True
queue: "GPUlongB"
project: "cenpes-lde"
n_workers: 4
cores: 1
processes: 1
memory: 2
shot_batch_size: 4
extra_job:
  - '-e slurm-%j.err'
  - '-o slurm-%j.out'
  - '--time=72:00:00'
  - '--requeue'
  - '--job-name="dask-modeling"'
```
To optimize the processing of a large number of shots, we divided them into batches. The number of shots per batch is set in `shot_batch_size` and is equal to the number of workers `n_workers`. If the local cluster configuration is used, the `memory` key represents the memory limit (in GB) per worker processes. On the contrary, it represents the total amount of memory (in GB) that the code could possibly use (not the memory per worker). For the local cluster configuration, only `n_workers` and `memory` parameters are of interest.  The keys `queue`, `project` and `extra_job` are meaningful only in the case of non-local cluster usage.

Finally, after the configuration is completed, execute the command:
```
python forward_script.py
```
Then, check the `shotfile_path` folder to see the generated segy files.

## FWI

The first thing we want to say is that we perform the inversion test on inverse-crime data, i.e. we use the same numerical
method for both generating the synthetic data and performing the inversion. However, once the inversion implementationn facilitates segy files reading, the inverse crime problem can be overcome. Just as in the previous case, it is also necessary to define some parameters in the YAML file (`config.yaml`). Clearly, the first step is to set the `forward` variable to `false`. Because the inversion code was designed to allow you to run either LSRTM or FWI you need to specific which will be run. For FWI, set `fwi` to `true`, otherwise (i.e. LSRTM), set it to `false`. Define bounding box constraints on the solution `vmin` and `vmax`, and choose one of the available gradient-based methods in `sotb-wrapper` library. You also need to inform the depth (number of layers) which the gradient must not be updated `mute_depth`. An example of the inversion parameter settings for the FWI experiment using the Marmousi II synthetic data is given below. Here we use the Limited Broyden-Fletcher-Goldfarb-Shanno method `opt_method: 'LBFGS'`
``` yaml 
forward: false
fwi: true
vmin: 1.377
vmax: 4.688
opt_meth: 'LBFGS'
mute_depth: 12
```
We maintain the cluster setup and modeling parameters. In the first case all is reused, while in the second one, a large part of the informations is used. You can also use checkpointing tecnhique (via [pyrevolve](https://github.com/devitocodes/pyrevolve)) to keep memory requirements as low as possible. In this case, simply enter the required values in each key in the dictionary `ckp_params`. To perform the experiment with or without the checkpointing technique, specify a boolean value (true/false) in the first key `checkpointing`, in according with the case. You can enter the percentage of total number of time steps to be checkpoints `pct_ckp`. If you choose a `null` value for this key, the number of checkpoints is set to be 1% (0.01) of the total number of time steps. In the case of a multilevel checkpointing strategy, you have to decide how best to divide the number of checkpoints between disk and RAM memory. Althought `pyrevolve` supports checkpointing with multiple levels of storage, we assume that only these two levels are available. Then, the percentage of checkpoints to be saved in memory `pct_mckp` has to be set. In this case, it is a percentage of the number of checkpoints defined by `pct_ckp`. Additionally you must set two lists  with the writing cost and the
reading cost of both levels of storage (`costs_mem` and `costs_disk`). With regard to the checkpoints who will be saved in memory, you have the option to save them on a single file or multiple files. To do this, set the key `singlefile` to `true` or `false`, respectively. To run the FWI experiment with the synthetic data from the Marmousi II model, on latop referred above (which has 16GB of RAM), we used the following configuration. 
``` yaml 
# checkpointing parameters
ckp_params:
  checkpointing: false
  ckptype: "MULTILEVEL_CKP"
  pct_ckp: null
  pct_mckp: 0.05
  singlefile: False
  costs_mem: [5, 4]
  costs_disk: [100, 50]
```
Since we do not have memory restrictions, we set `checkpointing: false`. By doing this, all other key values become irrelevant. If we wanted to run again the experiment, but using checkpointing, we simply had to set `checkpointing: true` and `pct_ckp: null`, because we are dealing with a 2D case, which does not merit a multilevel approach. Again, further key values become irrelevant as they are needed only for the multilevel case (i.e. `ckptype: "MULTILEVEL_CKP"`).

To run the FWI test, execute the command:
```
python inversion_script.py
```
After finishing the inversion process, you can generate the figures running the plot_fwi_results file.
```
python plot_fwi_results.py
```

## LSRTM

You must generate the linearized data before you can run the inversion experiments. Proceed as described in [Forward modeling](#forward-modeling) section and enter the required information into the YAML file to create the shot files. Keep in mind that in this case, you have to set the `born` key  in `solver_params` dictionary to `true`.

Once shot files are made available, update the `config.yaml`. In this case, both keys `forward` and `fwi` must be set to `false`. Also, choose the type of gradient-based method from `sotb-wrapper` library. As there are not now box constraints, the velocity limits are simply ignored.

Exactly like the previous case, you can decide whether or not to use the checkpointing technique. You must to define the checkpointing settings as explained in last section. Because the most of key-value pairs of the `solver_params` dictionary, which were used in generating the linearized data, will be used again, they are maintained as they stand in the YAML file.  The cluster setup also remains the same.

Use the inversion script to execute the LSRTM test:
```
python inversion_script.py
```





