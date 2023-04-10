#!/bin/bash
#SBATCH --job-name=air_sim
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:P100:1
#SBATCH --ntasks=1
#SBATCH --time=160:00:00
module load cuda
nvidia-smi
cd air_sim
nvcc kernel.cu -o kernel -std=c++11
./kernel