# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import sys

image_size = 28
hidden_units = 100
batch_size = 32
training_step = 100

ps_host = 'localhost:1001'
worker_host_1 = 'localhost:1101'
worker_host_2 = 'localhost:1102'

task_index = int(sys.argv[1])
# Is this uses as parameter server?
is_ps = sys.argv[2]
# Is used as ps OR worker?
job_name = str(sys.argv[3])

cluster = tf.train.ClusterSpec({'ps': [ps_host], 'worker':[worker_host_1, worker_host_2]})

server = tf.train.Server(cluster, task_index=task_index, job_name=job_name)

if is_ps:
    server.join()
else:
    # Assign ops to workers
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%task_index, cluster=cluster)):
        # Here is data and label placeholder
        x = tf.placeholder(tf.float32, [None, image_size * image_size])
        y = tf.placeholder(tf.float32, [None, 10])

        # Here is variables for w and b
        w = tf.Variable(tf.random_normal([image_size*image_size, hidden_units]))
        b = tf.Variable(tf.random_normal([hidden_units]))

        # Here is the logits of prediction
        logits = tf.add(tf.matmul(x, w), b)
        pred = tf.nn.softmax(logits)

        # This is the crossentropy loss
        crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)
        loss = tf.reduce_mean(crossentropy)

        # Here is accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, pred), tf.float32))

        global_step = tf.contrib.framework.get_or_create_global_step()

        optimizer = tf.train.GradientDescentOptimizer(.1).minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('Here is start for training steps!')
# This dose not use the general Session, here is monitor Session including is_chief
with tf.train.MonitoredSession(master=server.target, is_chief=(task_index == 0)) as sess:
    # Whether or not to stop the session
    while not sess.should_stop():
        sess.run(init)
        for i in range(training_step):
            batchx, batchy = mnist.train.next_batch(batch_size)

            _, l = sess.run([optimizer, logits], feed_dict={x: batchx, y: batchy})
            if i % 10 == 0:
                print('Step %d, loss=%.2f'%(i, l))

        print('All training finished!')
        print('-'*20)
        print('Now is used for model prediction:')

        xtest, ytest = mnist.test.images, mnist.test.labels
        prediction = sess.run(prediction, feed_dict={x: xtest})
        acc_test = sess.run(accuracy, feed_dict={x:xtest, y: ytest})
