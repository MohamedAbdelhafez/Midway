#!/bin/env python 


import numpy as np
import os,sys,inspect,time
import subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 


itera = 8

   
for ii in range(itera):
    time.sleep( np.random.random_sample())
    with open('out.txt', 'a') as the_file:
        the_file.write (str(ii) + " " + str(os.environ["SLURMD_NODENAME"]) + " " + str (time.time()) + "\n")
    
    