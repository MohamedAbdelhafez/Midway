#!/bin/bash

#SBATCH --job-name=arrayJob
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --time=00:01:00
#SBATCH --partition=sandyb
#SBATCH --mem-per-cpu=4000
#SBATCH --output=example_sbatch.out
#SBATCH --error=example_sbatch.err

module load python

python main.py --nodes $SLURM_JOB_NODELIST