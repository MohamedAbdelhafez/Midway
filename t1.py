#!/bin/env python 


import numpy as np
import os,sys,inspect
import subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 



print ("node_id : " + str(os.environ['SLURM_NODEID']))
print ("task_id : " + str(os.environ['SLURM_PROCID']))
print ("host_name : " + str(os.environ['HOSTNAME']))
print("-->" + repr(subprocess.check_output("hostname", universal_newlines=True, stderr=STDOUT)))
 