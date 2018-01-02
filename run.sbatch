#!/bin/bash

#SBATCH --job-name=gpuSbatch
#SBATCH --output=gpuSbatch.out
#SBATCH --error=gpuSbatch.err
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH -p gpu2
#SBATCH --gres=gpu:1

module load Anaconda3/5.0.0.1
moudle load cuda/8.0
python Python 3.6.2 |Anaconda custom (64-bit)| (default, Jul 20 2017, 13:51:32) main.py
