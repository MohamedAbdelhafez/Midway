#!/bin/env python 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import numpy, time
#import tensorflow as tf
#import os, sys
from time import sleep
from numpy.random import random_sample


import numpy as np
import os,sys,inspect


import tensorflow as tf
import scipy.linalg as la


import random as rd
import time
import math

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from time import sleep
from numpy.random import random_sample

#Defining time scales
total_time = 5
steps = 50
state_transfer = True
RWA = True
RFT = True

#Defining H0





qubit_state_num = 2

fq= 4.6/(2*np.pi)
kappa = 0.05
gamma = 0.001
g = 0.05

mode_state_num = 5
#g = 2.*np.pi*0.1 #GHz
fc = 5.0/(2*np.pi) #GHz
state_num = qubit_state_num * mode_state_num
if RFT:
    fq = fq-fc
    fc = 0
    
wc = 2*np.pi*fc
wa = 2*np.pi*fq


alpha = 0.224574
ens = np.array([ 2*np.pi*ii*(fq - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
H0q = np.kron(np.identity(mode_state_num),np.diag(ens))

a   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),1),np.identity(qubit_state_num))
adag   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),-1),np.identity(qubit_state_num))
sm = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),1))
smdag = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1))

if RWA:
     H0 = wc * np.dot(adag,a) + H0q + g * (np.dot(adag,sm) + np.dot(a,smdag))
else:
     H0 = wc * np.dot(adag,a) + H0q +  g * np.dot((adag + a),(sm + smdag))
#Defining Forbidden sates


#Defining Concerned states (starting states)
psi0=[0,1]

#Defining states to include in the drawing of occupation
states_draw_list = [0,1,2]
states_draw_names = ['g0','e0','g1']



U =[]
U1 = np.zeros(state_num,dtype=complex)
U1[1]=1
U1[0]=0
U.append(U1)
U2 = np.zeros(state_num,dtype=complex)
U2[0]=1
U.append(U2)
    

#Defining U0 (Initial)
q_identity = np.identity(qubit_state_num)
U0= q_identity

#Defining control Hs
IX = a + adag
IY = (0+1j)* (a-adag)
Hops = [IX]
ops_max_amp = [0.05]
Hnames =['HI']

#Defining convergence parameters
max_iterations = 5000
decay = max_iterations/2
convergence = {'rate':0.005, 'update_step':10, 'max_iterations':max_iterations,\
               'conv_target':1e-6,'learning_rate_decay':decay}
reg_coeffs = {'envelope' : 0,  'dwdt':0,'d2wdt2':0}


c_ops=[]
c_ops.append(np.sqrt(gamma)*sm)
c_ops.append(np.sqrt(kappa)*a)

u0 = None




      
draw = [states_draw_list,states_draw_names]
                    
show_plots = False
initial_guess = u0
use_gpu = False,
unitary_error = 1e-4
maxA=ops_max_amp
method ='Adam'
expect_op = IX
file_name='JC'
trajectories = 300
do_all_traj = False,
data_path = str(currentdir)+'/Data'
save = False
use_inter_vecs=True



# start time
grape_start_time = time.time()
freq_unit = "GHz"
# set timing unit used for plotting
freq_time_unit_dict = {"GHz": "ns", "MHz": "us","KHz":"ms","Hz":"s"}
time_unit = freq_time_unit_dict[freq_unit]

# make sparse_{H,U,K} False if use_gpu is True, as GPU Sparse Matmul is not supported yet.
if use_gpu:
    sparse_H = False
    sparse_U = False
    sparse_K = False

file_path = None



if U0 is None:
    U0 = np.identity(len(H0))
if convergence is None:
    convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,'conv_target':1e-8,'learning_rate_decay':2500}


if maxA is None:
    if initial_guess is None:
        maxAmp = 4*np.ones(len(Hops))
    else:
        maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
else:
    maxAmp = maxA

Taylor_terms = None
if state_transfer:
    no_scaling = True
else:
    no_scaling = False
dressed_info = None
# pass in system parameters
#sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,psi0,dressed_info,maxAmp, draw,initial_guess,  show_plots,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms, use_gpu, use_inter_vecs,sparse_H,sparse_U,sparse_K, c_ops, trajectories, do_all_traj, expect_op)





def dense_to_one_hot(labels_dense, num_classes = 10) :
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def run_training(server, cluster_spec, num_workers, task_index) :
    is_chief = (task_index == 0 and job_name == "worker")
    with tf.Graph().as_default():        
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster = cluster)) :           
            
            global_step = tf.get_variable('global_step', [],
                initializer = tf.constant_initializer(0), trainable = False)

            # Create the model
            x = tf.placeholder("float", [None, 784])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.nn.softmax(tf.matmul(x, W) + b)

            # Define loss and optimizer
            y_ = tf.placeholder("float", [None, 10])
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            opt = tf.train.GradientDescentOptimizer(0.01)
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate = num_workers,
                 total_num_replicas = num_workers)
            train_step = opt.minimize(cross_entropy, global_step = global_step)
            sync_replicas_hook = opt.make_session_run_hook(is_chief)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            init_token_op = opt.get_init_tokens_op()
            chief_queue_runner = opt.get_chief_queue_runner()

            init = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief = is_chief,
                init_op = init,
                
                recovery_wait_secs=0.1,
                global_step = global_step)
            # Create a session for running Ops on the Graph. device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
            config = tf.ConfigProto(allow_soft_placement = True, device_filters=['/job:ps', '/job:worker/task:%d' % task_index])
            #config = tf.ConfigProto(allow_soft_placement = True)
            sess = sv.prepare_or_wait_for_session(server.target, config = config)

            if is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])                
                sess.run(init_token_op)
                
            print ("Entering iterations: ")

            for i in range(20):
                if is_chief:
                    sleep(0.01)
                source_data = np.random.normal(loc = 0.0, scale = 1.0, size = (100, 784))
                labels_dense = np.clip(np.sum(source_data, axis = 1) / 5 + 5, 0, 9).astype(int)
                labels_one_hot = dense_to_one_hot(labels_dense)
                _, cost, acc, step = sess.run([train_step, cross_entropy, accuracy, global_step], feed_dict = { x: source_data, y_ : labels_one_hot })
                print("[%d]: cost=%.2f, accuracy=%.2f" % (step, cost, acc))
                with open('out.txt', 'a') as the_file:
                    the_file.write (str(i) + " " + str(os.environ["SLURMD_NODENAME"]) + " " + str (time.time()) + " " +str(step) + " " +str(acc) +"\n")
                #print(node_name)


    

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

idx = hosts.index(node_name+":2222") 
if (idx==0):
    job_name = "ps"
    task_index = 0
else:
    job_name = "worker"
    task_index = idx -1
cluster = tf.train.ClusterSpec( {"ps" : [hosts[0]], "worker": hosts[1:] } )
server = tf.train.Server(server_or_cluster_def=cluster,
                             job_name=job_name, task_index=task_index)




print(hosts)
print(node_name,job_name,task_index)
sys.stdout.flush()




if job_name == "ps":
    server.join()
else:
    run_training(server, cluster, len(hosts)-1, task_index)




