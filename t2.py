#!/bin/env python 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy, time
import tensorflow as tf
import os, sys

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
                
                recovery_wait_secs=1,
                global_step = global_step)
            # Create a session for running Ops on the Graph.
            config = tf.ConfigProto(allow_soft_placement = True, device_filters=[
    '/job:ps', '/job:worker/task:%d' % task_index])
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




