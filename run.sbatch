#!/bin/bash

#SBATCH --job-name=gpuSbatch
#SBATCH --output=gpuSbatch.out
#SBATCH --error=gpuSbatch.err
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=sandyb


module load Anaconda3/5.0.0.1

python main.py 