# fwi-lsrtm-python

This repository provides simple Python implementations of least-squares reverse time migration (LSRTM) and full-waveform inversion (FWI) for TTI- and (visco)acoustic
media. The implementations of both inversion workflows are based on the [Devito](https://www.devitoproject.org/), [sotb-wrapper](https://github.com/ofmla/seiscope_opt_toolbox_w_ctypes) and [Dask](https://dask.org/) libraries. The 2-D TTI Marmousi model, 2-D TTI Overthrust model and 
the 2-D TTI Marmousi II model are used to prove the robustness of the implementations.

## Forward modeling

For generating synthetic seismic data, you are first required to edit the YAML file in the `config` folder to change parameters for configuring the geometry and set the correct absolute paths in the variables `parfile_path` and `shotfile_path` of dictionary `solver_params`. They point to the folders were the physical model parameters are and shot files will be saved, respectively. Further control parameters need to be specified in the dictionary, including the minimum and maximum recording times `t0` and `tn`, sampling rate `dt`, peak frequency `f0`, order of accuracy of the finite difference scheme for spatial derivatives `space_order`, number of layers of the absorbing boundary condition `nbl` and floating-point precision to be used by Devito in code generation. In these settings you can choose from three modeling modes (`acoustic`, `viscoacoustic` and `tti`) to produce the seismic data. Set the `setup_func` to your preference. Born modeling can also be enabled by setting the `born` key to `true`. 

See below definitions in `config.yaml` to create data from the Marmousi II model. Note that both, sources and receivers are evenly distributed over the x-dimension. This means that you only need to specify the number for each of them `nshots` and `nrecs`, and their depths `src_depth` and `rec_depth`. 
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
