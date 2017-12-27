f = open('out.txt','w')
f.write(str($SLURM_JOB_NODELIST))
f.close()