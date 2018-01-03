#!/bin/bash

#SBATCH --job-name=gpuSbatch
#SBATCH --output=gpuSbatch.out
#SBATCH --error=gpuSbatch.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH -p bigmem

module load Anaconda3/5.0.0.1
module load cuda/8.0
which python
python -u main.py
