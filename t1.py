#!/bin/env python 

import os 
import subprocess 

print(os.environ["HOSTNAME"]) 
print("-->" + repr(subprocess.getstatusoutput("hostname")))