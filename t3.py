#!/bin/env python 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy, time
import tensorflow as tf
import os, sys

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
sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxAmp, draw,initial_guess,  show_plots,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms, use_gpu, use_inter_vecs,sparse_H,sparse_U,sparse_K, c_ops, trajectories, do_all_traj, expect_op)

def dense_to_one_hot(labels_dense, num_classes = 10) :
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def run_training(server, cluster_spec, num_workers, task_index) :
    is_chief = (task_index == 0)
    with tf.Graph().as_default():        
        with tf.device(tf.train.replica_device_setter(cluster = cluster_spec)) :            
            with tf.device('/cpu:0') :
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
                
                recovery_wait_secs=30,
                global_step = global_step)
            # Create a session for running Ops on the Graph.
            config = tf.ConfigProto(allow_soft_placement = True)
            sess = sv.prepare_or_wait_for_session(server.target, config = config)

            if is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])                
                sess.run(init_token_op)
            print ("Entering iterations: ")

            for i in range(10):
                source_data = numpy.random.normal(loc = 0.0, scale = 1.0, size = (100, 784))
                labels_dense = numpy.clip(numpy.sum(source_data, axis = 1) / 5 + 5, 0, 9).astype(int)
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









if job_name == "ps":
    server.join()
else:
    run_training(server, cluster, len(hosts)-1, task_index)




