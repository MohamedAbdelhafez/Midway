#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--nodes", help="assign nodes id for this run",default='0')
#args = parser.parse_args()
#nodes = str(args.nodes)
#start_node = int(nodes[7:10])
#end_node = int(nodes[11:14])
#f = open('out.txt','w')
#for ii in range (end_node - start_node):
    
    #f.write(str(start_node+ii) + " ")
    
#f.close()
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 
from main_grape.Grape_original import Grape

task_index  = int( os.environ['SLURM_PROCID'] )
n_tasks     = int( os.environ['SLURM_NPROCS'] )
tf_hostlist = os.environ['SLURM_NODELIST']

print (":P")
print (task_index)
print (":P")
print (n_tasks)
print (":P")
print (tf_hostlist)