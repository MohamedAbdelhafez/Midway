#!/bin/bash

#SBATCH --job-name=gpuSbatch
#SBATCH --output=gpuSbatch.out
#SBATCH --error=gpuSbatch.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module load tensorflow
python main.py 