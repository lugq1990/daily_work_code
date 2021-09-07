# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#define the params
learning_rate = .01
batch_size = 128
epochs = 10
display_epoch = 1
input_size = 784
n_classes = 10
dropout = .5
summary_path = 'E:\\tensorboard\\20170125\cnn'

x = tf.placeholder(tf.float32,[None,input_size],name='input_data')
y = tf.placeholder(tf.float32,[None,n_classes],name='label')
keep_prob = tf.placeholder(tf.float32,name='dropout')

weights = {'w1':tf.Variable(tf.random_normal([5,5,1,32],name='weight_1',dtype=tf.float32)),
           'w2':tf.Variable(tf.random_normal([5,5,32,64],name='wight_2',dtype=tf.float32)),
           'fc1':tf.Variable(tf.random_normal([7*7*64,1024],name='weight_fc1',dtype=tf.float32)),
           'out':tf.Variable(tf.random_normal([1024,n_classes],name='weight_out',dtype=tf.float32))}
biases = {'b1':tf.Variable(tf.random_normal([32],name='b_1',dtype=tf.float32)),
          'b2':tf.Variable(tf.random_normal([64],name='b_2',dtype=tf.float32)),
          'b_fc1':tf.Variable(tf.random_normal([1024],name='b_fc1',dtype=tf.float32)),
          'b_out':tf.Variable(tf.random_normal([n_classes],name='b_out',dtype=tf.float32))}

def con2d(x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
def pool(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],padding='SAME',strides=[1,k,k,1])

def conv_model(x,weights,biases,dropout):
    x = tf.reshape(x,[-1,28,28,1])
    #build the convolutional layer
    conv1 = con2d(x,weights['w1'],biases['b1'])
    conv1 = pool(conv1)
    conv2 = con2d(conv1,weights['w2'],biases['b2'])
    conv2 = pool(conv2)
    #next is the fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['fc1'].get_shape().as_list()[0]])
    # fc1 = tf.flatten(conv2)
    fc1 = tf.matmul(fc1,weights['fc1'])+biases['b_fc1']
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)
    out = tf.matmul(fc1,weights['out'])+biases['b_out']
    return out

with tf.name_scope('logits'):
    logits = conv_model(x,weights,biases,dropout)
with tf.name_scope('pred'):
    pred = tf.nn.softmax(logits)
with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
with tf.name_scope('correct_pred'):
    corrected_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(corrected_pred,tf.float32))

init = tf.global_variables_initializer()
#get the scaler data
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(summary_path,graph=tf.get_default_graph())
    for epoch in range(epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy = mnist.train.next_batch(batch_size)
            _,c,summary = sess.run([train_op,loss,merged_summary],feed_dict={x:batchx,y:batchy,keep_prob:dropout})
            summary_writer.add_summary(summary,global_step=epoch*total_batch+i)
        if(epoch % display_epoch)==0:
            print('epoch is ',epoch)
    print('optimization done!')
    test_data = mnist.test.images
    test_label = mnist.test.labels
    acc = sess.run(accuracy,feed_dict={x:test_data,y:test_label,keep_prob:1})
    print('test accuracy is ',acc)
