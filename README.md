# nnc_air
`nnc_air` is a software package that we release as supplementary material for the results in [[1]](#1). The package is capable of efficiently computing information rates of the noisy nanopore channel (NNC) [[1]](#1) for an arbitrary Markov source. The Markov source can be heuristically-optimised using the generalised Blahut-Arimoto algorithm (<a href="https://github.com/mcba1n/GBAA" target="_blank">GBAA</a>) [[2]](#2), [[3]](#3), which is included in this package.

This package is able to reproduce the results in [[1]](#1), and additionally the results in [[2]](#2), [[3]](#3).

## `air_sim`: Estimating information rates
### Choosing the model parameters
The NNC is specified by the channel mapping `f`, the adjacency matrix `A`, the Gaussian noise level `sigma`, and the duration distribution `P_K`. In this package, we choose `f` and `P_K` using CSV files in the directory `root/air_sim/name_of_model`. The adjacency matrix is specified by the Markov distribution `P`, which is also chosen using a CSV file in the same directory. The noise level `sigma` is chosen as a range of values in the `root/air_sim/kernel.cu` script, which we detail in the next section.

> To estimate information rates of an ISI channel with AWGN, make the contents of `P_K.csv` equal to `1`. Alternatively, see the later section on GBAA.

### Choosing the simulation parameters
The simulation parameters in the `root/air_sim/kernel.cu` script are:
- Block length `m`.
- `sigma_min`, `sigma_max`, and `delta` to simulate `sigma` for the range of values `sigma_min:delta:sigma_max`.
- Channel mapping `f` read from CSV `root/air_sim/name_of_model/f.csv`.
- Duration probabilities `P_K` read from CSV `root/air_sim/name_of_model/P_K.csv`.
- Markov distribution `P` read from CSV `root/air_sim/name_of_model/P.csv`.
- Intervals `intervals`. 
- The data is saved to `root/air_sim/data/data.csv`.

By default, the intervals are chosen to be those specified in [[1]](#1) using an epsilon `eps` equal to 1e-10 in the function `bd_int()`. Use the function `periodic_markers()` for computing information rates with markers. 

Note that any change in channel parameters or simulation parameters must be updated from within the script.

## `gbaa_sim`: Optimising the Markov source
The Markov source of the aforementioned NNC can be heuristically-optimised using GBAA on an NNC that has been synchronised, which is a finite-state channel (FSC). Specifically, the synchronised NNC is a faded ISI channel, with fading according to `P_K` (known at the receiver), and AWGN noise according to `sigma`. 

### Choosing the model parameters
The parameters `f` and `A` are specified using a `.mat` file (e.g., `root/gbaa_sim/scrappie_graph.mat`). The duration distribution `P_K` and noise level `sigma` are specified in the `root/gbaa_sim/gbaa_nnc_fading.m` script.

### Choosing the simulation parameters
The simulation parameters in the `root/gbaa_sim/gbaa_nnc_fading.m` script are:

- Block length `m`.
- The range of values for `sigma` in `sigma_vals`.
- The duration distribution `P_K`.
- The maximum number of iterations `max_iters` by GBAA.
- The early stopping condition parameter `eps`. 
If the information rates in consecutive iterations differ less than `eps`, then GBAA is stopped.
- The data is saved to `root/gbaa_sim/optimised_P.mat`.


## Running
The following is a step-by-step guide for running the scripts `root/air_sim/kernel.cu` and `root/gbaa_sim/gbaa_nnc_fading.m`. When using complex models in the NNC, such as the nanopore in `air_sim/nanopore_model/`, we recommend running these scripts on a cluster. Hence, the following guides are based on a Linux environment.

> For a cluster using the Slurm workload manager, run the job scripts `air_sim.sh` and `gbaa_sim.sh` using `sbatch`. Otherwise, follow the guides below.

### air_sim
The script `root/air_sim/kernel.cu` is a so-called CUDA kernel that can only run on an NVIDIA GPU with CUDA installed.

1. Install CUDA (e.g., CUDA Version 11.7).
2. Ensure that the `root` directory is correctly specified within the script for each of the aforementioned CSV files.
3. Run the commands:
```
cd air_sim
nvcc kernel.cu -o kernel -std=c++11
./kernel
```

### gbaa_sim
The script `root/gbaa_sim/gbaa_nnc_fading.m` is a standard MATLAB program that should run on an any relatively new version of MATLAB.

1. Install MATLAB (e.g., MATLAB R2021a).
2. Run the commands:
```
export MATLABROOT=/usr/local/matlab/r2021a
cd gbaa_sim
mcc -mv gbaa_nnc_fading.m -o gbaa_nnc_fading -a ./GBAA
echo 10 | ./run_gbaa_nnc_fading.sh  $MATLABROOT
```
Alternatively, you may use the MATLAB GUI.

## References
<a id="1">[1]</a> 
McBain, B., Viterbo, E., Saunderson, J. (2023). 
Information rates of the noisy nanopore channel.
IEEE Transactions on Information Theory (submitted).

<a id="2">[2]</a> 
Kavcic, A. (2001). 
On the capacity of Markov sources over noisy channels.
GLOBECOM'01. IEEE Global Telecommunications Conference.

<a id="3">[3]</a> 
Vontobel, P., Kavcic, A., Arnold, D.-M., and Loeliger, H.-A. (2008). 
A generalized Blahut-Arimoto algorithm.
IEEE Transactions on Information Theory.
