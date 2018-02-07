from multiprocessing import Process
import tensorflow as tf
from time import sleep
from numpy.random import random_sample
import os

cluster = tf.train.ClusterSpec({'ps': ['localhost:2222'],
                                'worker': ['localhost:2223',
                                           'localhost:2224',
                                           'localhost:2225',
                                           'localhost:2226']})


def create_worker(task_index):
    server = tf.train.Server(cluster, job_name='worker', task_index=task_index)

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
        nasty_var = tf.Variable(0)  # This line causes the problem. No issue when this is commented out.

    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_index == 0)):
        for step in xrange(10000):
            sleep(random_sample())  # Simulate some work being done.
            print 'Worker %d | step %d' % (task_index, step)


def create_ps(task_index):
    param_server = tf.train.Server(cluster, job_name='ps',
                                   task_index=task_index)
    param_server.join()

    
    
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
# Launch workers and ps in separate processes.
processes = []
for i in xrange(len(cluster.as_dict()['worker'])):
    print 'Forking worker process ', i
    p = Process(target=create_worker(task_index), args=[i])
    p.start()
    processes.append(p)