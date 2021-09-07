# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
path = 'E:\\tensorboard\\20180306'

learning_rate = .01
batch_size = 128
epochs = 100
display_step = 1

hidden_1 = 256
hidden_2 = 256

x = tf.placeholder(tf.float32,[None,784],name='data')
y = tf.placeholder(tf.float32,[None,10],name='label')

weights = {'w1':tf.Variable(tf.random_normal([784,hidden_1]),name='w1'),
     'w2':tf.Variable(tf.random_normal([hidden_1,hidden_2]),name='w2'),
     'out':tf.Variable(tf.random_normal([hidden_2, 10]),name='out')}
biases = {'b1':tf.Variable(tf.zeros([hidden_1]),name='b1'),
          'b2':tf.Variable(tf.zeros([hidden_2]),name='b2'),
          'b_out':tf.Variable(tf.zeros([10]),name='b_out')}

def MLP(x,weights,biases):
    layer1 = tf.add(tf.matmul(x,weights['w1']), biases['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['w2']), biases['b2'])
    out = tf.add(tf.matmul(layer2,weights['out']),biases['b_out'])
    return out

with tf.name_scope('logits'):
    logits = MLP(x,weights,biases)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar('accuracy',accuracy)
tf.summary.scalar('loss',loss)

summery_merge_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    loss_list = list()
    acc_list = list()
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        summary_writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
        for i in range(total_batch):
            batchx,batchy = mnist.train.next_batch(batch_size)
            _,c,acc,summary = sess.run([optimizer,loss,accuracy,summery_merge_op],feed_dict={x:batchx,y:batchy})
            avg_cost += c/total_batch
        if epoch % display_step == 0:
            print('epoch = {0:d}, loss = {1:.7f}'.format(epoch, avg_cost))
            loss_list.append(avg_cost)
            acc_list.append(acc)
            summary_writer.add_summary(summary,epoch*batch_size+i)

    print('done!')
    test_data,test_label = mnist.test.images,mnist.test.labels
    acc_test = sess.run(accuracy,feed_dict={x:test_data,y:test_label})
    print('test accuracy = {0:.9f}'.format(acc_test))

    #plot the loss and accuracy curve
    fig = plt.figure(figsize=(8,6))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_list, label='loss')
    fig2 = plt.figure(figsize=(8,6))
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(acc_list)
    plt.show()

