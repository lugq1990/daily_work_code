# -*- coding:utf-8 -*-
"""This is just used for showing how to use tensorflow distribution for Distributing training"""
import tensorflow as tf
import sys

host1 = 'localhost:1002'
host2 = 'localhost:1003'

# Get which task to process from terminal
task_num = int(sys.argv[1])

# Make a cluster with localhost for 2 web addresses
cluster = tf.train.ClusterSpec({'local': [host1, host2]})
# Here is used to build a server, task_index is used for which task to be placed on.
server = tf.train.Server(cluster, job_name='local', task_index=task_num)

print('Starting server :{}'.format(task_num))

sess = tf.Session(server.target)

a = tf.constant(2)

### This is not to specify which operation to run on which place.
# b = a - 10
# c = a + 12
# out = b + c
# print('*'*20, sess.run(out))


### Here I can also specify which variable running place
with tf.device('/job:local/task:0'):
    b = a - 10

with tf.device('/job:local/task:1'):
    c = a + 12
    out = b + c

with tf.Session('grpc://'+ host1) as sess:    ## Here can also use some service to run the session
    print('*'*20)
    print('Final result:', sess.run(out))   ## Result is also with be printed on task1 for host2


# Start the server and join the service
server.start()
server.join()


# import tensorflow as tf
# c = tf.constant("Hello, distributed TensorFlow!")
# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)  # Create a session on the server.
# sess.run(c)



