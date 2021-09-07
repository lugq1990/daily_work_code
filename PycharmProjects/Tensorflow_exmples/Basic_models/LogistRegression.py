# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#hyperpameters
learning_rete = .01
batch_size = 128
epochs = 10
display_step = 1

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.add(tf.matmul(x,w),b))

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
#cost = tf.losses.softmax_cross_entropy(y, pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rete).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_list = list()
    accuracy_list = list()
    validation_list = list()
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx, batchy = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost],feed_dict={x:batchx,y:batchy})
            avg_cost += c/total_batch
        loss_list.append(avg_cost)
        accuracy_list.append(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
        validation_list.append(sess.run(accuracy, feed_dict={x:mnist.validation.images, y:mnist.validation.labels}))
        if epoch % display_step == 0:
            print('epoch = {0:d}, loss = {1:.9f}'.format(epoch,avg_cost))

    print('done!')
    test_data,test_label = mnist.test.images,mnist.test.labels
    print('test accuracy is',sess.run(accuracy,feed_dict={x:test_data,y:test_label}))

    #plot the loss curve
    fig = plt.figure(figsize=(10,8))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.plot(loss_list, label='loss')

    fig2 = plt.figure(figsize=(10,8))
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Train and validation curve')
    plt.plot(accuracy_list, label='train_accuracy')
    plt.plot(validation_list, label='validation_accuracy')
    plt.legend()
    plt.show()
