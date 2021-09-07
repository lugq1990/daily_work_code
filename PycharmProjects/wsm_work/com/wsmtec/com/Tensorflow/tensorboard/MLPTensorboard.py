# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#params
learning_rate = .01
batch_size = 128
n_input = 784
n_classes = 10
epochs = 10
display_epoch = 1
dropout = .5
n_hidden_1 = 1024
n_hidden_2 = 512
path = 'E:\\tensorboard\\20170125\mlp'

#placeholder of data and label
x = tf.placeholder(tf.float32,[None,n_input],name='data')
y = tf.placeholder(tf.float32,[None,n_classes],name='label')
keep_ratio = tf.placeholder(tf.float32,name='dropout')

weights = {'fc1':tf.Variable(tf.random_normal([n_input,n_hidden_1],dtype=tf.float32),name='fc1'),
           'fc2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],dtype=tf.float32),name='fc2'),
           'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],dtype=tf.float32),name='out')}
biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1],dtype=tf.float32),name='b1'),
          'b2':tf.Variable(tf.random_normal([n_hidden_2],dtype=tf.float32),name='b2'),
          'b_out':tf.Variable(tf.random_normal([n_classes],dtype=tf.float32),name='b_out')}

with tf.name_scope('fc1'):
    layer1 = tf.matmul(x,weights['fc1'])+biases['b1']
with tf.name_scope('drop1'):
    drop1 = tf.nn.dropout(layer1,keep_prob=keep_ratio)
with tf.name_scope('fc2'):
    layer2 = tf.matmul(drop1,weights['fc2'])+biases['b2']
with tf.name_scope('drop2'):
    drop2 = tf.nn.dropout(layer2,keep_prob=keep_ratio)
with tf.name_scope('logits'):
    logits = tf.matmul(drop2,weights['out'])+biases['b_out']
with tf.name_scope('pred'):
    pred = tf.nn.softmax(logits)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
with tf.name_scope('train_op'):
    train_op = optimizer.minimize(loss)
with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

summary_merged_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(path,graph=tf.get_default_graph())
    for epoch in range(epochs):
        total_batch = int(mnist.train.num_example)