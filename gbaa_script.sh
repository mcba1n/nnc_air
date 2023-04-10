#!/bin/bash
#SBATCH --job-name=gbaa_sim
#SBATCH --qos=normal
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --time=150:00:00

module load  matlab/r2021a

MCR_CACHE_ROOT=$TMPDIR
export MCR_CACHE_ROOT

export MATLABROOT=/usr/local/matlab/r2021a

cd gbaa_sim
mcc -mv gbaa_nnc_fading.m -o gbaa_nnc_fading -a ./GBAA
echo 10 | ./run_gbaa_nnc_fading.sh  $MATLABROOT
