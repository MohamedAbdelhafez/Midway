#!/bin/env python 


import numpy as np
import os,sys,inspect
import subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 


itera = 8

for ii in range(itera):
    print (ii)
    print(str(os.environ["SLURMD_NODENAME"]))
    sys.stdout.flush()
    