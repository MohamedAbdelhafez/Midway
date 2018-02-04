import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import scipy.linalg as la
from core.TensorflowState import TensorflowState
from core.SystemParameters import SystemParameters
#from core.Convergence import Convergence
#from core.run_session import run_session

import sys
import random as rd
import time
import math
from helper_functions.grape_functions import c_to_r_mat, sort_ev
from core.RegularizationFunctions import get_reg_loss
from tensorflow.python.framework import function
from tensorflow.python.framework import ops


def get_inner_product(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for single vector
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num])
    psi_1_imag = (psi1[state_num:2*state_num])
    psi_2_real = (psi2[0:state_num])
    psi_2_imag = (psi2[state_num:2*state_num])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.multiply(psi_1_real,psi_2_real)
        bd = tf.multiply(psi_1_imag,psi_2_imag)
        bc = tf.multiply(psi_1_imag,psi_2_real)
        ad = tf.multiply(psi_1_real,psi_2_imag)
        reals = tf.square(tf.add(tf.reduce_sum(ac),tf.reduce_sum(bd)))
        imags = tf.square(tf.subtract(tf.reduce_sum(bc),tf.reduce_sum(ad)))
        norm = tf.add(reals,imags)
    return norm

def get_loss_list(sys_para,psi1,psi2):
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude

    ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
    bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
    bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
    ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
    ac_bd = tf.square(tf.add(ac,bd))
    bc_ad = tf.square(tf.subtract(bc,ad))

    loss_list = tf.add(ac_bd,bc_ad)
    return loss_list

def get_inner_product_2D(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors
    # psi1 and psi2 are shaped as (2*state_num, number of vectors)
    state_num=sys_para.state_num


    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)

        ac_bd = tf.square(tf.add(ac,bd))
        bc_ad = tf.square(tf.subtract(bc,ad))
        reals = tf.reduce_sum(ac_bd) # first trace inner product of all vectors, then squared
        imags = tf.reduce_sum(bc_ad)
        norm = (tf.add(reals,imags))/(tf.cast(num_vecs,tf.float32))
    return norm

def get_inner_product_3D(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors and timesteps
    # psi1 and psi2 are shaped as (2*state_num, time_steps, number of vectors)
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
        reals = tf.reduce_sum(tf.square(tf.reduce_sum(tf.add(ac,bd),1)))
        # first trace inner product of all vectors, then squared, then sum contribution of all time steps
        imags = tf.reduce_sum(tf.square(tf.reduce_sum(tf.subtract(bc,ad),1)))
        norm = (tf.add(reals,imags))/(len(sys_para.states_concerned_list)**2)
    return norm

def get_avgd_inner_product ( sys_para, psi1, psi2, start, end):
    state_num=sys_para.state_num


    psi_1_real = (psi1[0:state_num,start:end])
    psi_1_imag = (psi1[state_num:2*state_num,start:end])
    psi_2_real = (psi2[0:state_num,start:end])
    psi_2_imag = (psi2[state_num:2*state_num,start:end])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)

        ac_bd = tf.add(ac,bd)
        bc_ad = tf.subtract(bc,ad)
        reals = tf.reduce_sum(ac_bd)/tf.cast((end-start), tf.float32) # first trace inner product of all vectors, then squared
        imags = tf.reduce_sum(bc_ad)/tf.cast((end-start), tf.float32)

    return reals, imags

def expect (sys_para, num_trajs,  op, psis):
    result = []
    psis2 = tf.matmul(tf.cast(op,tf.float32),psis)
    if num_trajs[0] !=0:

        expect1 = get_avgd_inner_product (sys_para, psis, psis2, 0, num_trajs[0])
        if not sys_para.do_all:
            result.append(expect1)
    else:
        expect1 = 0
    if num_trajs[1] !=0:
        expect2 = get_avgd_inner_product (sys_para, psis, psis2, num_trajs[0], num_trajs[0] + num_trajs[1])
        if not sys_para.do_all:
            result.append(expect2)
    else:
        expect2 = 0
    if sys_para.do_all:
        return expect1, expect2
    else:
        return tf.stack(result)

def normalize(sys_para, psi, num_vecs):
    state_num=sys_para.state_num
    new_norms = tf.reshape(get_norms(sys_para, psi, num_vecs),[num_vecs])
    weights = 1/tf.sqrt(new_norms)
    x = []
    for ii in range (2*state_num):
        x.append(weights)
    return tf.multiply(psi,tf.stack(x))




def get_norms(sys_para, psi, num_vecs):
    state_num=sys_para.state_num
    psi1 = tf.reshape(psi,[2*state_num,num_vecs])
    return tf.reduce_sum(tf.square(psi1),0)

def get_norm(sys_para, psi):
    state_num=sys_para.state_num
    psi1 = tf.reshape(psi,[2*state_num,1])
    return tf.reduce_sum(tf.square(psi1),0)
def get_one_random(sys_para, num_trajs, start,end,index):
    vec_type = tf.constant(0)
    sums = []
    s = 0
    for jj in range (len(sys_para.initial_vectors)):
        #create a list of their summed probabilities
        s=s+num_trajs[jj]
        sums=tf.concat([sums,tf.reshape(s,[1])],0)

    r2 = tf.cast(index,tf.int32)
    rvector=r2 * tf.ones_like(sums)
    cond2= tf.greater_equal(sums,rvector)
    b=tf.where(cond2)
    final =tf.reshape(b[0,:],[])
    return tf.random_uniform([1],tf.gather(start,final),tf.gather(end,final))



def get_random(sys_para,num_trajs,  start,end,length=1):

    #Returns a random number between 0 & 1 to tell jumps when to occur
    ii =0
    rand = []
    for initial_vector in sys_para.initial_vectors:
        new = tf.random_uniform([num_trajs[ii]],start[ii],end[ii])
        if rand == []:
            rand = new
        else:
            rand = tf.concat([rand,new],0)
        ii = ii+1

    #rand=tf.random_uniform([length],start,end)
    return rand

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
    sys.stdout.flush()
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

    print ("Building graph:")
    sys.stdout.flush()
    if job_name == "ps":
        print ("PS Joined")
        sys.stdout.flush()
        server.join()
    else:
        print ("Worker running")
        sys.stdout.flush()
        is_chief = task_index == 1
        with tf.Graph().as_default():  
            with tf.device(tf.train.replica_device_setter(cluster = cluster)) :            
                with tf.device('/cpu:0') :
            

                    input_num = len(sys_para.Hnames) +1
                    taylor_terms = sys_para.exp_terms 
                    scaling = sys_para.scaling
                    if sys_para.traj:
                        tf_c_ops = tf.constant(np.reshape(sys_para.c_ops_real,[len(sys_para.c_ops),2*sys_para.state_num,2*sys_para.state_num]),dtype=tf.float32)
                        tf_cdagger_c = tf.constant(np.reshape(sys_para.cdaggerc,[len(sys_para.c_ops),2*sys_para.state_num,2*sys_para.state_num]),dtype=tf.float32)
                        norms=[]
                        jumps=[]
                        if sys_para.expect:
                            expect_op = tf.constant(sys_para.expect_op)


                    def get_matexp(uks,H_all):
                        # matrix exponential
                        I = H_all[input_num]
                        matexp = I
                        uks_Hk_list = []
                        for ii in range(input_num):
                            uks_Hk_list.append((uks[ii]/(2.**scaling))*H_all[ii])

                        H = tf.add_n(uks_Hk_list)
                        H_n = H
                        factorial = 1.

                        for ii in range(1,taylor_terms+1):      
                            factorial = factorial * ii
                            matexp = matexp + H_n/factorial
                            if not ii == (taylor_terms):
                                H_n = tf.matmul(H,H_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_U)

                        for ii in range(scaling):
                            matexp = tf.matmul(matexp,matexp,a_is_sparse=sys_para.sparse_U,b_is_sparse=sys_para.sparse_U)

                        return matexp


                    @function.Defun(tf.float32,tf.float32,tf.float32)
                    def matexp_op_grad(uks,H_all, grad):  
                        # gradient of matrix exponential
                        coeff_grad = []

                        coeff_grad.append(tf.constant(0,dtype=tf.float32))


                        ### get output of the function
                        matexp = get_matexp(uks,H_all)          
                        ###

                        for ii in range(1,input_num):
                            coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                                   tf.matmul(H_all[ii],matexp,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_U))))

                        return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32)]                                         

                    global matexp_op


                    @function.Defun(tf.float32,tf.float32, grad_func=matexp_op_grad)                       
                    def matexp_op(uks,H_all):
                        # matrix exponential defun operator
                        matexp = get_matexp(uks,H_all)

                        return matexp 

                    def get_matvecexp(uks,H_all,psi):
                        # matrix vector exponential
                        I = H_all[input_num]
                        matvecexp = psi

                        uks_Hk_list = []

                        for ii in range(input_num):
                            uks_Hk_list.append(uks[ii]*H_all[ii])

                        H = tf.add_n(uks_Hk_list)    

                        psi_n = psi
                        factorial = 1.

                        for ii in range(1,taylor_terms):      
                            factorial = factorial * ii
                            psi_n = tf.matmul(H,psi_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K)
                            matvecexp = matvecexp + psi_n/factorial

                        return matvecexp


                    @function.Defun(tf.float32,tf.float32,tf.float32,tf.float32)
                    def matvecexp_op_grad(uks,H_all,psi, grad):  
                        # graident of matrix vector exponential
                        coeff_grad = []

                        coeff_grad.append(tf.constant(0,dtype=tf.float32))

                        ### get output of the function
                        matvecexp = get_matvecexp(uks,H_all,psi)
                        #####


                        for ii in range(1,input_num):
                            coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                                   tf.matmul(H_all[ii],matvecexp,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K))))



                        I = H_all[input_num]
                        vec_grad = grad
                        uks_Hk_list = []
                        for ii in range(input_num):
                            uks_Hk_list.append((-uks[ii])*H_all[ii])

                        H = tf.add_n(uks_Hk_list)
                        vec_grad_n = grad
                        factorial = 1.

                        for ii in range(1,taylor_terms):      
                            factorial = factorial * ii
                            vec_grad_n = tf.matmul(H,vec_grad_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K)
                            vec_grad = vec_grad + vec_grad_n/factorial

                        return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32),vec_grad]                                         

                    global matvecexp_op

                    @function.Defun(tf.float32,tf.float32,tf.float32, grad_func=matvecexp_op_grad)                       
                    def matvecexp_op(uks,H_all,psi):
                        # matrix vector exponential defun operator
                        matvecexp = get_matvecexp(uks,H_all,psi)

                        return matvecexp





                    tf_one_minus_gaussian_envelope = tf.constant(sys_para.one_minus_gauss,dtype=tf.float32, name = 'Gaussian')

                    num_vecs = len(sys_para.initial_vectors)

                    if sys_para.traj:
                        tf_initial_vectors=[]
                        num_trajs = tf.placeholder(tf.int32, shape = [num_vecs])
                        vecs = tf.reshape(tf.constant(sys_para.initial_vectors[0],dtype=tf.float32),[1,2*sys_para.state_num])
                        targets = tf.reshape(tf.constant(sys_para.target_vectors[0],dtype =tf.float32),[1,2*sys_para.state_num])
                        ii = 0
                        counter = tf.constant(0)
                        for initial_vector in sys_para.initial_vectors:

                            tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
                            target_vector = tf.reshape(tf.constant(sys_para.target_vectors[ii],dtype =tf.float32),[1,2*sys_para.state_num])
                            tf_initial_vectors.append(tf_initial_vector)
                            tf_initial_vector = tf.reshape(tf_initial_vector,[1,2*sys_para.state_num])
                            i = tf.constant(0)

                            c = lambda i,vecs,targets: tf.less(i, num_trajs[ii])

                            def body(i,vecs,targets):

                                def f1(): return tf.concat([vecs,tf_initial_vector],0), tf.concat([targets,target_vector],0)
                                def f2(): return tf_initial_vector, target_vector
                                vecs,targets = tf.cond(tf.logical_and(tf.equal(counter,tf.constant(0)),tf.equal(i,tf.constant(0))), f2, f1)



                                return [tf.add(i,1), vecs,targets]

                            r,vecs,targets = tf.while_loop(c, body, [i,vecs,targets],shape_invariants = [i.get_shape(), tf.TensorShape([None,2*sys_para.state_num]), tf.TensorShape([None,2*sys_para.state_num])])
                            counter = tf.add(counter,r)
                            ii = ii+1
                        vecs = tf.transpose(vecs)
                        targets = tf.transpose(targets)
                        packed_initial_vectors = vecs
                        target_vecs = targets
                        num_vecs = counter

                    else:
                        tf_initial_vectors=[]
                        for initial_vector in sys_para.initial_vectors:
                            tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
                            tf_initial_vectors.append(tf_initial_vector)
                        packed_initial_vectors = tf.transpose(tf.stack(tf_initial_vectors))

                    H0_weight = tf.Variable(tf.ones([sys_para.steps]), trainable=False) #Just a vector of ones needed for the kernel
                    weights_unpacked=[H0_weight] #will collect all weights here
                    ops_weight_base = tf.Variable(tf.constant(sys_para.ops_weight_base, dtype = tf.float32), dtype=tf.float32,name ="weights_base")

                    ops_weight = tf.sin(ops_weight_base,name="weights")
                    for ii in range (sys_para.ops_len):
                        weights_unpacked.append(sys_para.ops_max_amp[ii]*ops_weight[ii,:])

                    #print len(sys_para.ops_max_amp)
                    H_weights = tf.stack(weights_unpacked,name="packed_weights")



                    print ("Operators weight initialized.")
                    sys.stdout.flush()


                
                    global_step = tf.get_variable('global_step', [], 
                                      initializer = tf.constant_initializer(0), 
                                      trainable = False,
                                      dtype = tf.int32)


                    jump_vs = []
                    tf_matrix_list = tf.constant(sys_para.matrix_list,dtype=tf.float32)
                    # Create a trajectory for each initial state
                    Evolution_states=[]
                    inter_vecs=[]
                    inter_lst = []
                    #start = tf.placeholder(tf.float32,shape=[])
                    #end = tf.placeholder(tf.float32,shape=[])
                    start = tf.placeholder(tf.float32,shape=[len(sys_para.initial_vectors)])
                    end = tf.placeholder(tf.float32,shape=[len(sys_para.initial_vectors)])
                    psi0 = packed_initial_vectors
                    old_psi = psi0
                    new_psi = psi0
                    norms = tf.ones([num_vecs],dtype = tf.float32)
                    r=get_random(sys_para,num_trajs,  start,end,num_vecs)
                    operator = tf_c_ops[0] # temporary
                    expects = []
                    inter_vecs_list=[]
                    inter_vecs_list.append(old_psi)
                    all_jumps= []
                    all_norms = []
                    all_norms.append(norms)
                    vecs = tf.cast(num_vecs, tf.int64)
                    for ii in np.arange(0,sys_para.steps):
                        old_psi = new_psi        
                        new_psi = matvecexp_op(H_weights[:,ii],tf_matrix_list,old_psi)
                        new_norms = tf.reshape(get_norms(sys_para, new_psi, num_vecs),[num_vecs])

                        norms = tf.multiply(norms,new_norms)
                        all_norms.append(norms)

                        cond= tf.less(norms,r)
                        a=tf.where(cond)
                        state_num=sys_para.state_num
                        reshaped_new = tf.reshape(new_psi,[2*state_num*num_vecs])

                        c = tf.constant(0)
                        def while_condition(c,old,new,norms,randoms):
                            return tf.less(c, tf.size(a))
                        def jump_fn(c,old,new,norms,randoms):


                            index = tf.reshape(tf.gather(a,c),[])
                            idx = []

                            for kk in range (2*state_num):
                                idx.append(index + kk*vecs)

                            vector = tf.gather(reshaped_new,idx)
                            #vector = tf.gather(tf.transpose(old),index)


                            #####


                            if len(sys_para.c_ops)>1:
                                weights=[]
                                sums=[]
                                s=0
                                for ii in range (len(sys_para.c_ops)):

                                    temp=tf.matmul(tf.transpose(tf.reshape(vector,[2*state_num,1])),tf_cdagger_c[ii,:,:])
                                    temp2=tf.matmul(temp,tf.reshape(vector,[2*state_num,1])) #get the jump expectation value
                                    weights=tf.concat([weights,tf.reshape(temp2,[1])],0)
                                weights=tf.abs(weights/tf.reduce_sum(tf.abs(weights))) #convert them to probabilities

                                for jj in range (len(sys_para.c_ops)):
                                    #create a list of their summed probabilities
                                    s=s+weights[jj]
                                    sums=tf.concat([sums,tf.reshape(s,[1])],0)

                                r2 = tf.random_uniform([1],0,1)
                                #tensorflow conditional graphing, checks for the first time a summed probability exceeds the random number
                                rvector=r2 * tf.ones_like(sums)
                                cond2= tf.greater_equal(sums,rvector)
                                b=tf.where(cond2)
                                final =tf.reshape(b[0,:],[])
                                #final = tf.gather(b,0)

                                #apply the chosen jump operator
                                propagator2 = tf.reshape(tf.gather(tf_c_ops,final),[2*sys_para.state_num,2*sys_para.state_num])
                            else:
                                propagator2 = tf.reshape(tf_c_ops,[2*sys_para.state_num,2*sys_para.state_num])
                            inter_vec_temp2 = tf.matmul(propagator2,tf.reshape(vector,[2*sys_para.state_num,1]))
                            norm2 = get_norm(sys_para, inter_vec_temp2)
                            inter_vec_temp2 = inter_vec_temp2 / tf.sqrt(norm2)

                            #delta = tf.reshape(inter_vec_temp2 - tf.gather(tf.transpose(new),index),[2*sys_para.state_num])

                            new_vector = tf.reshape(tf.gather(tf.reshape(new,[2*state_num*num_vecs]),idx),[2*sys_para.state_num])
                            inter_vec_temp2 = tf.reshape(inter_vec_temp2,[2*sys_para.state_num])
                            #delta = inter_vec_temp2 
                            delta = inter_vec_temp2-new_vector
                            indices=[]
                            for jj in range (2*sys_para.state_num):
                                indices.append([jj,index])

                            values = delta
                            shape = tf.cast(tf.stack([2*sys_para.state_num,num_vecs]),tf.int64)
                            Delta = tf.SparseTensor(indices, values, shape)
                            new = new + tf.sparse_tensor_to_dense(Delta)


                            values = tf.reshape(1 - tf.gather(norms,index),[1])
                            shape = tf.cast(tf.stack([num_vecs]),tf.int64)
                            Delta_norm = tf.SparseTensor(tf.reshape(index,[1,1]), values, shape)
                            norms = norms + tf.sparse_tensor_to_dense(Delta_norm)

                            #new_random = get_one_random(start, end,index)
                            new_random =tf.random_uniform([1],0,1)
                            values = tf.reshape(new_random - tf.gather(randoms,index),[1])
                            #shape = tf.stack([num_vecs])
                            Delta_norm = tf.SparseTensor(tf.reshape(index,[1,1]), values, shape)
                            randoms = randoms + tf.sparse_tensor_to_dense(Delta_norm)

                            #####

                            return [tf.add(c, 1),old,new,norms,randoms]

                        wh,old_psi,new_psi,norms,r = tf.while_loop(while_condition, jump_fn, [c,old_psi,new_psi,norms,r])
                        all_jumps.append(wh)


                        new_psi = normalize(sys_para, new_psi, num_vecs)

                        inter_vecs_list.append(new_psi)
                        if sys_para.expect:

                            expects.append(expect(sys_para, num_trajs, expect_op, new_psi))

                    inter_vecs_packed = tf.stack(inter_vecs_list, axis=1)
                    inter_vecs = inter_vecs_packed
                    all_norms = tf.stack(all_norms)
                    if sys_para.expect:
                        if sys_para.do_all:
                            expectations = tf.stack(expects, axis=1)
                        else:
                            expectations = tf.stack(expects)
                    else:
                        expectations = 0

                    #####

                    #inter_vecs_packed.set_shape([2*sys_para.state_num,sys_para.steps,num_vecs] )
                    #inter_vecs2 = tf.unstack(inter_vecs_packed, axis = 2)
                    #indices = tf.stack(indices)






                    #inter_vec = tf.reshape(psi0,[2*sys_para.state_num,1],name="initial_vector")
                    #psi0 = inter_vec


                    all_jumps = tf.stack(all_jumps)
                    jumps.append(jumps)
                    #jumps = tf.stack(jumps)
                    #for tf_initial_vector in tf_initial_vectors:
                        #Evolution_states.append(One_Trajectory(tf_initial_vector)) #returns the final state of the trajectory
                    packed = inter_vecs_packed
                    print ("Trajectories Initialized")
                    sys.stdout.flush()


                    if sys_para.state_transfer == False:

                        final_vecs = tf.matmul(final_state, packed_initial_vectors)

                        loss = 1-get_inner_product_2D(sys_para, final_vecs,target_vecs, num_vecs)

                    else:
                        #loss = tf.constant(0.0, dtype = tf.float32)
                        final_state = inter_vecs_packed[:,sys_para.steps,:]
                        a = []
                        for ii in range (sys_para.steps):
                            a.append(tf.constant((sys_para.steps-ii), dtype = tf.float32))
                        accelerate = tf.stack(a)
                        accelerate = tf.ones([sys_para.steps])
                        #
                        if sys_para.expect:

                            Il1 = tf.reduce_sum(expectations[:,0,0])  
                            Il2 = -tf.reduce_sum(expectations[:,1,0])
                            Il = Il1 + Il2
                            Il1d = tf.gradients(Il1, [ops_weight_base])[0]
                            Il2d = tf.gradients(Il2, [ops_weight_base])[0]
                            loss = - tf.square(Il)
                            quad = tf.gradients(loss, [ops_weight_base])[0]




                        unitary_scale = get_inner_product_2D(sys_para, final_state,final_state, num_vecs)


                    reg_loss = loss

                    print ("Training loss initialized.")
                    sys.stdout.flush()
                    learning_rate = tf.placeholder(tf.float32,shape=[])
                    opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
                    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(hosts)-1,
                                            total_num_replicas=len(hosts)-1)
                    sync_replicas_hook = opt.make_session_run_hook(is_chief)

                    #Here we extract the gradients of the pulses
                    grad = opt.compute_gradients(reg_loss)

                    grad_pack = tf.stack([g for g, _ in grad])
                    var = [v for _,v in grad]

                    grads =[tf.nn.l2_loss(g) for g, _ in grad]
                    grad_squared = tf.reduce_sum(tf.stack(grads))


                    gradients =[g for g, _ in grad]
                    avg_grad = tf.placeholder(tf.float32, shape = [1,len(sys_para.ops),sys_para.steps])

                    new_grad = zip(tf.unstack(avg_grad),var)
                    #new_grad = grad

                    if sys_para.traj:
                        #optimizer = opt.apply_gradients(new_grad, global_step = global_step)
                        #optimizer = opt.apply_gradients(grad, global_step = global_step)
                        optimizer = opt.minimize(reg_loss, global_step = global_step)
                        
                        
                    else:
                        optimizer = opt.apply_gradients(grad)


                    #optimizer = opt.apply_gradients(grad)

                    print ("Optimizer initialized.")
                    sys.stdout.flush()





                    print ("Graph " +str(task_index) + " built!")
                    sys.stdout.flush()
                init_op = tf.global_variables_initializer()
                init_token_op = opt.get_init_tokens_op()
                chief_queue_runner = opt.get_chief_queue_runner()


                sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir="/tmp",

                                     init_op=init_op,
                                     recovery_wait_secs=20,
                                     global_step=global_step)

                sess = sv.prepare_or_wait_for_session(server.target) 

                itera = 0
                if is_chief:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)


                traj_num = sys_para.trajectories
                max_traj = 1000
                num_psi0 = len(sys_para.initial_vectors)
                needed_traj = []
                for kk in range (num_psi0):
                    needed_traj.append(traj_num)
                jump_traj = np.sum(needed_traj)
                num_batches = len(hosts)-1
                num_traj_batch = int(traj_num/num_batches)
                lrate = 0.005
                fd_dict = {learning_rate: lrate, start: np.zeros([num_psi0]), end: np.ones([num_psi0]), num_trajs:num_traj_batch*np.ones([num_psi0])}
                print ("Entering iterations_"+str(task_index))
                sys.stdout.flush()
                for ii in range(5):

                    my_print('\r'+' Iteration: ' +str(ii) + ": Running batch #" +str(task_index+1)+" out of "+str(num_batches)+ " with "+str(num_traj_batch)+" jump trajectories")
                    #sys.stdout.flush()

                    #nos, exs, l1d,l2d,  q, l1, l2, int_vecs = sess.run([norms, expectations, Il1d, Il2d,quad, Il1, Il2, inter_vecs], feed_dict=fd_dict)
                    _, step = sess.run([optimizer, global_step], feed_dict=fd_dict)
                    #print (np.square(l1 + l2))
                    #sys.stdout.flush()
                    my_print (ii)
                    my_print(task_index)
                    #time.sleep( np.random.random_sample())
                    with open('out.txt', 'a') as the_file:
                        the_file.write('\r'+' Iteration: ' +str(ii) + ": Running batch #" +str(task_index+1)+" out of "+str(num_batches)+ " with "+str(num_traj_batch)+" jump trajectories " + "step:" + str(step) +  "\n")

                    #sys.stdout.flush()


    #conv = Convergence(sys_para,time_unit,convergence)
def my_print(text):
    sys.stdout.write(str(text)+'\n')
    sys.stdout.flush()
    # run the optimization
    #SS = run_session(tfs,graph,conv,sys_para,method, show_plots = sys_para.show_plots, use_gpu = use_gpu)
    #return SS.uks,SS.Uf
