import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from core.SystemParameters import SystemParameters
#from core.Convergence import Convergence
#from core.run_session import run_session


import random as rd
import time

import os

def divide(needed,max_traj):
    returned = []

    end = False

    while not end:
        summation = 0
        trial = np.zeros_like(needed)
        flag = True

        for ii in range (len(needed)):
            if flag and ((summation + needed[ii]) <= max_traj):
                summation = summation + needed[ii]
                trial[ii] = needed[ii]
                if ii == len(needed)-1:
                    end = True
            else:

                trial[ii] = max_traj - summation
                summation = max_traj
                flag = False
        returned.append(trial)
        needed = needed-trial
    return returned
    
def Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence = None, U0= None, reg_coeffs = None,dressed_info = None, maxA = None ,use_gpu= True, sparse_H=True,sparse_U=False,sparse_K=False,draw= None, initial_guess = None,show_plots = True, unitary_error=1e-4, method = 'Adam',state_transfer = False,no_scaling = False, freq_unit = 'GHz', file_name = None, save = True, data_path = None, Taylor_terms = None, use_inter_vecs=True, c_ops = None, trajectories = 500, do_all_traj = False, expect_op = []):
    
    # start time
    grape_start_time = time.time()
    
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
    
    # pass in system parameters
    sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxAmp, draw,initial_guess,  show_plots,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms, use_gpu, use_inter_vecs,sparse_H,sparse_U,sparse_K, c_ops, trajectories, do_all_traj, expect_op)
    
    #run_python = Python_evolve(sys_para)

        
    
    tfs = TensorflowState(sys_para) # create tensorflow graph
    graph = tfs.build_graph()
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    if not use_gpu:
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        config = tf.ConfigProto(gpu_options = gpu_options)
    with graph.as_default():
        sess = tf.train.MonitoredTrainingSession(master=tfs.server.target, config = config, is_chief=tfs.is_chief,
                                             hooks=[tfs.sync_replicas_hook])
        itera = 0

        traj_num = sys_para.trajectories
        max_traj = 1000
        num_psi0 = len(sys_para.initial_vectors)
        needed_traj = []
        for kk in range (num_psi0):
            needed_traj.append(traj_num)
        jump_traj = np.sum(needed_traj)
        num_batches = len(tfs.hosts)-1
        num_traj_batch = int(traj_num/num_batches)
        print ("entering iterations")
        for ii in range(convergence['max_iterations']):
            learning_rate = float(convergence['rate']) * np.exp(-float(ii) / convergence['learning_rate_decay'])
            print('\r'+' Iteration: ' +str(ii) + ": Running batch #" +str(tfs.task_index+1)+" out of "+str(num_batches)+ " with "+str(num_traj_batch)+" jump trajectories")
            sys.stdout.flush()
            feed_dict = {tfs.learning_rate: 0, self.tfs.start: np.zeros([num_psi0]), self.tfs.end: np.ones([num_psi0]), self.tfs.num_trajs:num_traj_batch*np.ones([num_psi0])}
            norms, expects, l1d,l2d,  quad, l1, l2, inter_vecs = sess.run([tfs.norms, tfs.expectations, tfs.Il1d, tfs.Il2d,tfs.quad, tfs.Il1, tfs.Il2, tfs.inter_vecs], feed_dict=self.feed_dict)

        

    #conv = Convergence(sys_para,time_unit,convergence)
    
    # run the optimization
    #SS = run_session(tfs,graph,conv,sys_para,method, show_plots = sys_para.show_plots, use_gpu = use_gpu)
    #return SS.uks,SS.Uf
        