# Memory profiling of 2D forward modeling using Devito

This repository provides simple Python implementations of 2D forward modeling for acoustic media. The implementations are based on the [Devito](https://www.devitoproject.org/) and [Dask](https://dask.org/) libraries. The [h5py](https://github.com/h5py/h5py), [segyio](https://github.com/equinor/segyio) packages are also used. The 2-D Marmousi model is used in the experiments. The main goal here is to monitor the memory during the execution of the modeling using the [memory-profile](https://pypi.org/project/memory-profiler/) tool.

## Forward modeling

You can generate synthetic seismic data in parallel or sequential way. Aditionally, you can choose between to use the `AcousticWaveSolver` Class from the Devito's folder `examples` or a Devito `Operator`. Thus, there are four possible options for generating the shot gathers:

1. Forward modeling for all the shots in sequential using the `AcousticWaveSolver` Class
2. Forward modeling for all the shots in sequential using a Devito `Operator`
3. Forward modeling for all the shots in parallel in a dask cluster using the `AcousticWaveSolver` Class
4. Forward modeling for all the shots in parallel in a dask cluster using a Devito `Operator`

The parameters used in the modeling are defined in a YAML file (`config.yaml`) in the `config` folder. They include parameters for configuring the geometry and the absolute paths of the folders where the physical parameters (in [hdf5](https://www.hdfgroup.org/solutions/hdf5) format) will be read and generated shots (in [segy](https://wiki.seg.org/wiki/SEG-Y) format) will be stored. These paths are set in the variables `parfile_path` and `shotfile_path` of the dictionary `solver_params`. Further control parameters are specified in the dictionary, including the minimum and maximum recording times `t0` and `tn`, sampling rate `dt`, peak frequency `f0`, order of accuracy of the finite difference scheme for spatial derivatives `space_order`, number of layers of the absorbing boundary condition `nbl` and floating-point precision to be used by Devito in code generation. In these settings you can decide whether or not to use an instance of the `AcousticWaveSolver` to produce the seismic data. Specify a boolean value (true/false) in the `use_solver` key, in according with the case.

To run in parallel you need set up the dask cluster in `config.yaml`. Depending on available hardware, you can launch a local Dask cluster (on your laptop or single computing node) or a non-local one (on a distributed HPC with job scheduler installed). In latter case we only support an HPC cluster which uses the SLURM scheduler. Set `use_local_cluster` to `false` or `true`, depending of the hardware. Here are the settings for running the forward modeling of 184 shots for the Marmousi model, on a quad core laptop (i.e., Intel(R) Core(TM) i7-8565U CPU @1.80GHz)

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

Finally, after the configuration is completed, you can profile each of the options provided in the above list using `GNU Make`. To do so, you can run `make` with the specific target for each option. For first and second options, use 
```
make serial.solver
```
and
```
make serial.operator
```
In the case of the third and fourth options
```
make dask.solver
```
and
```
make dask.operator
```

In each case, check the `shotfile_path` folder to see the generated segy files. After finishing the modeling process, you can plot the recorded memory usage running the `mprof plot` command for each file generated. You should obtain figures such as the following:


In the parallel case is possible to see a remarkable increase in the memory for both cases, when `Operator` as `AcousticWaveSolver` class are used. Although, each dask task (dedicated to the modleing of one shot) creates its own `Operator` or `solver` according to the case, I did not expect for a such increasing because theoretically the memory shoul be released after each function call. In the sequential case, the results are somewhat curious. When the `Operator` is used, the memory is almost constant over time, but when an `AcousticWaveSolver` instance is created and the class method `forward` is called repeatedly, the level of memory increases significantly. Since the memory comsumption was not approximately the same in both cases, it is possible that the `AcousticWaveSolver` class is not working as expected. Of course, there is always the possibility of I am doing the things in the wrong way.
