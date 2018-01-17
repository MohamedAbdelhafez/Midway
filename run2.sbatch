#!/bin/bash

#SBATCH --job-name=cpuSbatch
#SBATCH --time=00:20:00
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=1-4
#SBATCH --ntasks=16
#SBATCH -p broadwl

module load Anaconda3/5.0.0.1
module load cuda/8.0
which python
source activate DL_CPU
python -u main2.py
