# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import time
start_time = time.time()

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

path = 'E:\\tensorboard\\20180301'
learning_rate = .001
batch_size = 128
epochs = 100
display_step = 1

n_input = 28
n_hidden = 128
n_classes = 10
timestemps = 28

x = tf.placeholder(tf.float32,[None,timestemps,n_input],name='data')
y = tf.placeholder(tf.float32,[None,n_classes],name='label')

weights = {'out':tf.Variable(tf.random_normal([n_hidden,n_classes]),name='weights')}
biases = {'out':tf.Variable(tf.constant(.1,shape=[n_classes]),name='biases')}

#define the RNN structure
def RNN(x,weights,biases):
    x = tf.unstack(x,timestemps,1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.)
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.add(tf.matmul(outputs[-1],weights['out']),biases['out'])

logits = RNN(x,weights,biases)
with tf.name_scope('pred'):
    pred = tf.nn.softmax(logits=logits)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

#define the merge op
merge_all_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    loss_list = list()
    acc_list = list()
    eval_list = list()
    summary_writer = tf.summary.FileWriter(path,graph=tf.get_default_graph())
    for epoch in range(epochs):
        avg_loss = .0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy = mnist.train.next_batch(batch_size)
            batchx = batchx.reshape((batch_size,timestemps,n_input))
            _,l,summary = sess.run([optimizer,loss,merge_all_op],feed_dict={x:batchx,y:batchy})
            avg_loss += l/total_batch
            summary_writer.add_summary(summary,global_step=epoch*total_batch+i)
        if epoch % display_step==0 or epoch==0:
            acc_e = sess.run(accuracy,feed_dict={x:mnist.test.images.reshape((-1,timestemps,n_input)),
                                                        y:mnist.test.labels})
            val_acc = sess.run(accuracy,feed_dict={x:mnist.validation.images.reshape((-1,timestemps,n_input)),
                                                   y:mnist.validation.labels})
            loss_list.append(avg_loss)
            acc_list.append(acc_e)
            eval_list.append(val_acc)
            print('epoch={0:d},loss={1:.7f},accuracy={2:.7f}'.format(epoch,avg_loss,acc_e))
    print('done!')

    #evaluate the test data
    test_data, test_labels = mnist.test.images.reshape((-1,timestemps,n_input)),mnist.test.labels
    print('test accurcy is ',sess.run(accuracy,feed_dict={x:test_data,y:test_labels}))
    print('use '+np.str((time.time()-start_time)/60)+' minutes')

    #plot the result
    fig = plt.figure(figsize=(8,6))
    plt.title('loss curve')
    plt.plot(loss_list)
    fig2 = plt.figure(figsize=(8,6))
    plt.title('test and validate accuracy')
    plt.ylim(0,1)
    plt.plot(acc_list)
    plt.plot(eval_list)
    plt.show()
