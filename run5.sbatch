#!/bin/bash 

#SBATCH --job-name=cpuSbatch 
#SBATCH --output=cpuSbatch.out 
#SBATCH --error=cpuSbatch.err 
#SBATCH --time=02:00:00 
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --exclusive 
#SBATCH -p broadwl 
#SBATCH --tasks-per-node=1 



module load Anaconda3/5.0.0.1 
which python
source activate DL_CPU

chmod u+x test.py
unset http_proxy 
unset https_proxy
srun -n 1 -N 1 test.py  & 
srun -n 1 -N 1 test.py  &
srun -n 1 -N 1 test.py  &
srun -n 1 -N 1 test.py 




