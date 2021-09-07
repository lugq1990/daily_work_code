# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

path = 'E:\\tensorboard\\20170119'
learning_rate = .001
batch_size = 128
steps = 2000
display_step = 100
num_input = 28
times = 28
num_hidden = 128
num_classes = 10

x = tf.placeholder(tf.float32,[None,times,num_input])
y = tf.placeholder(tf.float32,[None,num_classes])

with tf.name_scope("weights"):
    weights = {'out':tf.Variable(tf.random_normal([2*num_hidden,num_classes]))}
with tf.name_score("biases"):
    biases = {'out':tf.Variable(tf.random_normal([num_classes]))}

def BiRNN(x,weights,biases):
    x = tf.unstack(x,times,1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.)
    try:
        outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,
                                                   x,dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,
                                               x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']

logits = BiRNN(x,weights,biases)
with tf.name_scope("pred"):
    pred = tf.nn.softmax(logits=logits)

#define the loss and optimizer
with tf.name_scope("loss"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

#evaluate the model
with tf.name_scope("correct_pred"):
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#init
init = tf.global_variables_initializer()
#define the scaler for the tensorboard
tf.summary.scalar('loss',loss_op)
tf.summary.scalar('accuracy',accuracy)
merged = tf.summary.merge_all()

#start training the model
with tf.Session() as sess:
    sess.run(init)
    #define the summary write for the sess graph
    summary_writer = tf.summary.FileWriter(path,graph=sess.graph)
    for step in range(1,steps+1):
        batchx,batchy = mnist.train.next_batch(batch_size)
        batchx = batchx.reshape((batch_size,times,num_input))
        sess.run(optimizer,feed_dict={x:batchx,y:batchy})
        if step % display_step==0:
            loss,acc,summary = sess.run([loss_op,accuracy,merged],feed_dict={x:batchx,y:batchy})
            # print('loss=',loss)
            # print('acc =',acc)
            print("step is "+np.str(step)+" and the loss = "+"{:.4f}".format(loss)+" and the accuracy is "+"{:.4f}".format(acc))
            summary_writer.add_summary(summary,step)
    print("optimizer done!")

    test_data = mnist.test.images.reshape((-1,times,num_input))
    test_label = mnist.test.labels
    print("test accuracy = ",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
    tf.summary.FileWriter(path+'/test')