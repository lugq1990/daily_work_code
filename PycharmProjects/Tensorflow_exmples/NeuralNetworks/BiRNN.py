# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import time

start_time = time.time()
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

path = 'E:\\tensorboard\\20180301'

#hyperparameters
learning_rate = .001
batch_size = 128
epochs = 1000
display_step = 1

n_input = 28
n_classes = 10
timestampes = 28
n_hidden = 128

#placeholder
x = tf.placeholder(tf.float32,[None,timestampes,n_input],name='data')
y = tf.placeholder(tf.float32,[None,n_classes],name='label')

#weights and biases
weights = {'out':tf.Variable(tf.random_normal([2*n_hidden,n_classes]),name='weights')}
biases = {'out':tf.Variable(tf.constant(.1,shape=[n_classes]),name='biases')}

#define the birnn
def BiRNN(x,weights,biases):
    x = tf.unstack(x,timestampes,1)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.)

    #get the lstm outputs
    try:
        outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)

    #return the outputs of lstm matmul the weights and added the bias
    return tf.matmul(outputs[-1],weights['out'])+biases['out']

#get the bidirectional outputs
logits = BiRNN(x,weights,biases)

with tf.name_scope('pred'):
    pred = tf.nn.softmax(logits=logits)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#init all variables
init = tf.global_variables_initializer()

#summary the scaler and histogram
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

#merge all the scaler
merge_all_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    loss_list = list()
    acc_list = list()
    eval_list = list()

    #define the file writer of the summary
    summary_writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
    for epoch in range(epochs):
        avg_loss = .0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy = mnist.train.next_batch(batch_size)
            batchx = batchx.reshape((batch_size,timestampes,n_input))
            _,summary,l = sess.run([optimizer,merge_all_op,loss],feed_dict={x:batchx,y:batchy})
            avg_loss += l/total_batch
        if epoch % display_step==0 or epoch == 0:
            eval_data,eval_label = mnist.validation.images.reshape((-1,timestampes,n_input)),mnist.validation.labels
            test_data,test_label = mnist.test.images.reshape((-1,timestampes,n_input)), mnist.test.labels
            eval_t = sess.run(accuracy,feed_dict={x:eval_data,y:eval_label})
            acc_t = sess.run(accuracy,feed_dict={x:test_data,y:test_label})
            loss_list.append(avg_loss)
            acc_list.append(acc_t)
            eval_list.append(eval_t)
            print('epoch={0:d},loss={1:.7f},test_acc={2:.7f},eval_acc={3:.7f}'.format(epoch,avg_loss,acc_t,eval_t))
            summary_writer.add_summary(summary,epoch*batch_size+i)

    print('done!')

    #evaluate the model on the test data
    print('the final test accuracy=',sess.run(accuracy,feed_dict={x:mnist.test.images.reshape((-1,timestampes,n_input)),
                                                                  y:mnist.test.labels}))
    print('the total procetrue us '+np.str((time.time()-start_time)/60)+' minutes')

    #plot the loss curve and accuracy curve
    fig = plt.figure(figsize=(8,6))
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_list)
    fig2 = plt.figure(figsize=(8,6))
    plt.title('test and validation curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0,1.0)
    plt.plot(acc_list)
    plt.plot(eval_list)
    plt.show()