# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

learning_rate = .01
epochs = 20
batch_size = 128
display_epoch = 1
path = 'E:/tensorboard/20170119/basic'

x = tf.placeholder(tf.float32,[None,784],name='input_data')
y = tf.placeholder(tf.float32,[None,10],name='label')

w = tf.Variable(tf.random_normal([784,10],stddev=.1),name='weight')
b = tf.Variable(tf.constant(shape=[10],value=.1),name='biase')

with tf.name_scope('pred'):
    pred = tf.nn.softmax(tf.matmul(x,w)+b)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.name_scope('accuracy'):
    acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',acc)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(path,graph=tf.get_default_graph())
    for epoch in range(epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy = mnist.train.next_batch(batch_size)
            _,c,summary = sess.run([optimizer,loss,merged_summary_op],feed_dict={x:batchx,y:batchy})
            summary_writer.add_summary(summary,epoch*total_batch+i)
        if epoch % display_epoch == 0:
            print('epochs is ',epoch)
    print('optimization done!')
    test_data = mnist.test.images
    test_label = mnist.test.labels
    accuracy = sess.run(acc,feed_dict={x:test_data,y:test_label})
    print('test data acc is ',"{:.5f}".format(accuracy))

