#!/bin/env python 

import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 

S_g = []
S_e = []

gIs = np.transpose(np.concatenate((np.load("gtrajs_10ns.npy"),np.load("gtrajs_10ns_1.npy")), axis = 1))
eIs = np.transpose(np.concatenate((np.load("etrajs_10ns.npy"),np.load("etrajs_10ns_1.npy")), axis = 1))
g_traj = np.mean(gIs, axis =0)
e_traj = np.mean(eIs, axis = 0)
diff = -(g_traj - e_traj)
ratio = 1
num_traj = len(gIs)
steps = int(ratio*len(gIs[0]))
num_train = int(3*num_traj/4)
g_train = gIs[0:num_train ]
e_train = eIs[0:num_train ]
g_test = gIs[num_train :]
e_test = eIs[num_train:]

g_labels = []
e_labels = []
for ii in range (num_traj):
    
    g_labels.append([1,0])
    e_labels.append([0,1])

g_train_labels = g_labels[0:num_train ]
e_train_labels = e_labels[0:num_train ]
g_test_labels = g_labels[num_train: ]
e_test_labels = e_labels[num_train: ]




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
else:
    job_name = "worker"
    task_index = idx -1
cluster = tf.train.ClusterSpec( {"ps" : [hosts[0]], "worker": hosts[1:] } )
server = tf.train.Server(server_or_cluster_def=cluster,
                         job_name=job_name, task_index=task_index)
if job_name == "ps":
    print ("PS Joined")
    server.join()
    
else:
    print ("Worker running")
    is_chief = task_index == 0
    
    with tf.device(tf.train.replica_device_setter( worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
        rate = 0.5
        
        W = tf.Variable(tf.zeros([steps, 2]))
        b = tf.Variable(tf.zeros([2]))
        m = tf.Variable(tf.zeros([steps,2]))
    
        global_step = tf.Variable(0, name="global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, steps])
        xt = tf.placeholder(tf.float32, [None, steps])
        yt = tf.matmul(tf.square(xt),m) + tf.matmul(xt, W) + b
        y = tf.matmul(tf.square(x),m) + tf.matmul(x, W) + b
        y_ = tf.placeholder(tf.float32, [None, 2])
        yt_ = tf.placeholder(tf.float32, [None, 2])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        opt = tf.train.AdamOptimizer(rate)
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(hosts)-1,
                                total_num_replicas=len(hosts)-1)
        train_step = opt.minimize(loss, global_step = global_step)
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        correct_prediction = tf.equal(tf.argmax(yt, 1), tf.argmax(yt_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init_op = tf.global_variables_initializer()
        init_token_op = opt.get_init_tokens_op()
        chief_queue_runner = opt.get_chief_queue_runner()
        print("---Variables initialized---")

        
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % task_index])
    
    
    sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                     
                                     init_op=init_op,
                                     #summary_op=summary_op,
                                     summary_op=None,
                                     recovery_wait_secs =0.1,
                                     logdir="/tmp",
                                     saver=tf.train.Saver(),
                                     global_step=global_step,
                                     save_model_secs=60)
    step = 0
    #sess = tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,
                                         #hooks=[sync_replicas_hook], config = sess_config)
     
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
        
    def batch_generator(ii, num_batches,  g_train, e_train, g_train_labels, e_train_labels, g_test, e_test, g_test_labels, e_test_labels):
        len_train_batch = int(len(g_train)/num_batches)
        len_test_batch = int(len(g_test)/num_batches)
        start_train = ii*len_train_batch
        end_train = (ii+1)*len_train_batch
        start_test = ii*len_test_batch
        end_test = (ii+1)*len_test_batch
        all_train = np.concatenate((g_train[start_train:end_train],e_train[start_train:end_train]))
        all_train_labels = np.concatenate((g_train_labels[start_train:end_train],e_train_labels[start_train:end_train]))
        all_test = np.concatenate((g_test[start_test:end_test],e_test[start_test:end_test]))
        all_test_labels = np.concatenate((g_test_labels[start_test:end_test],e_test_labels[start_test:end_test]))
        return all_train, all_train_labels, all_test, all_test_labels
    
    iterations = 100
    num_batches = len(hosts)-1
    print ("entering iterations")
    
    with sv.prepare_or_wait_for_session(server.target, config =sess_config) as sess:
        if task_index == 0:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
    
        for i in range(iterations):
            tr,trl,tes,tesl = batch_generator(task_index, num_batches,  g_train, e_train, g_train_labels, e_train_labels, g_test, e_test, g_test_labels, e_test_labels)



            all_train, all_train_labels = unison_shuffled_copies(tr, trl)
            all_test, all_test_labels = unison_shuffled_copies(tes, tesl)
            #sess.run(train_step, feed_dict={x: all_train, y_: all_train_labels})

            #print("Iteration "+ str(i))
            sys.stdout.flush()
            _,l,ac  = (sess.run([train_step, loss,accuracy], feed_dict={x: all_train,
                                              y_: all_train_labels, xt: all_test, yt_: all_test_labels}))





            step += 1
            print(step, task_index, l, ac, time.time())
            sys.stdout.flush()

    
    
