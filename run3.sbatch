#!/bin/bash 

#SBATCH --job-name=cpuSbatch 
#SBATCH --output=cpuSbatch.out 
#SBATCH --error=cpuSbatch.err 
#SBATCH --time=00:03:00 
#SBATCH --nodes=2 
#SBATCH --ntasks=2 
#SBATCH --exclusive 
#SBATCH -p broadwl 
#SBATCH --tasks-per-node=1 



module load Anaconda3/5.0.0.1 
chmod u+x t1.py
srun -n 1 -N 1 t1.py  & 
srun -n 1 -N 1 t1.py 