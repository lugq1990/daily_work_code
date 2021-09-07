# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
# from tensorflow.keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

host1 = 'localhost:1128'
host2 = 'localhost:1129'

num_epochs = 100

task_num = int(sys.argv[1])


# make a cluster
cluster = tf.train.ClusterSpec({'local': [host1, host2]})
# make a server
server = tf.train.Server(cluster, job_name='local', task_index=task_num)


# load the mnist datasets and normalize data
data = input_data.read_data_sets('MNIST', one_hot=True)


# define the model variables and model structure
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# get prediction
logits = tf.matmul(x, w) + b
pred = tf.nn.softmax(logits)

# get the loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# make the optimizer
optimizer = tf.train.GradientDescentOptimizer(.1).minimize(loss)

correct_pred = tf.equal(y, pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# create a session based on the serser sites
sess = tf.Session(target=server.target)
sess.run(init)


# loop for the num_epochs for training
batch_size = 16
for i in range(num_epochs):
    xbatch, ybatch = data.train.next_batch(batch_size)
    acc, l, _ = sess.run([accuracy, loss, optimizer], feed_dict={x: xbatch, y: ybatch})
    if i % 10 == 0:
        print('Now is step %d '%i)
        print('Accuracy=%.2f, loss=%.2f'%(acc, l))

server.start()
server.join()

