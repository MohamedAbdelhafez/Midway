#!/bin/bash

#SBATCH --job-name=cpuSbatch
#SBATCH --output=cpuSbatch.out
#SBATCH --error=cpuSbatch.err
#SBATCH --time=00:20:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH -p broadwl

module load Anaconda3/5.0.0.1
module load cuda/8.0
which python
source activate DL_CPU
python -u main2.py
