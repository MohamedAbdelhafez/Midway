#!/bin/env python 
import numpy as np
import os,sys,inspect, subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 



#print ("node_id : " + str(os.environ['SLURM_NODEID']))
#print ("task_id : " + str(os.environ['SLURM_PROCID']))
#print ("host_name : " + str(os.environ['HOSTNAME']))
#print("-->" + repr(subprocess.check_output("hostname")))


#task_index  = int( os.environ['SLURM_PROCID'] )
tf_hostlist = str(os.environ['SLURM_NODELIST'])
node_name = str(os.environ["SLURMD_NODENAME"])
foo = ( [pos for pos, char in enumerate(tf_hostlist) if char == ','])
clusters = tf_hostlist.count(',')
start_i = 9
hosts =[]
server = tf_hostlist[0:8]
for ii in range (clusters+1):
    if ii>0:
        start_i = foo[ii-1]+1
    if clusters==0:
        end_i = len(tf_hostlist)-1
    elif ii != clusters:
        end_i = foo[ii]
    else:
        end_i = len(tf_hostlist)-1
    string = tf_hostlist[start_i:end_i]
    if (len(string) <5):
        hosts.append(server+string+":2222")
    else:
        s = int(string[0:4])
        e = int(string[6:10])
        for jj in range (e-s+1):
            hosts.append(server+str(s+jj).zfill(4)+":2222")
print(hosts)
print(node_name)
idx = hosts.index(node_name+":2222") 
if (idx==0):
    job_name = "ps"
    task_index = 0
    print ("1st")
else:
    job_name = "worker"
    task_index = idx -1
    print ("rest")


    
    
