# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

path = 'E:\\tensorboard\\20180301'
#hyperparameters
learning_rate = .001
batch_size = 128
num_steps = 200
display_step = 10
dropout = .75

x = tf.placeholder(tf.float32,[None, 784],name='data')
y = tf.placeholder(tf.float32,[None,10],name='label')
keep_prob = tf.placeholder(tf.float32,name='dropout')

weights = {'wc1':tf.Variable(tf.random_normal([5,5,1,32]),name='wc1'),
           'wc2':tf.Variable(tf.random_normal([5,5,32,64]),name='wc2'),
           'wd1':tf.Variable(tf.random_normal([7*7*64,1024]),name='wd1'),
           'out':tf.Variable(tf.random_normal([1024,10]),name='w_out')}
biases = {'bc1':tf.Variable(tf.random_normal([32]),name='bc1'),
          'bc2':tf.Variable(tf.random_normal([64]),name='bc2'),
          'bd1':tf.Variable(tf.random_normal([1024]),name='bd1'),
          'out':tf.Variable(tf.random_normal([10]),name='b_out')}

def con2d(x,w,b,stride=1):
    with tf.name_scope('conv'):
        x = tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='SAME')
        x = tf.nn.bias_add(x,b)
        return tf.nn.relu(x)

def pooling(x,k=2):
    with tf.name_scope('pool'):
        return tf.nn.max_pool(x,ksize=[1,k,k,1],padding='SAME',strides=[1,k,k,1])

#create the model
def conv_net(x,weights,biases,dropput):
    x = tf.reshape(x,shape=[-1,28,28,1])
    conv1 = con2d(x,weights['wc1'],biases['bc1'])
    conv1 = pooling(conv1)
    conv2 = con2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = pooling(conv2)
    fc1 = tf.reshape(conv2,shape=[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    #fc1 = tf.layers.batch_normalization(fc1)
    fc1 = tf.nn.relu(fc1)
    fc = tf.nn.dropout(fc1,keep_prob=dropput)
    out = tf.add(tf.matmul(fc,weights['out']),biases['out'])
    return out

logits = conv_net(x,weights,biases,dropput=keep_prob)
with tf.name_scope('pred'):
    pred = tf.nn.softmax(logits)

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
merged_all = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    loss_list = list()
    acc_list = list()
    val_list = list()
    # for epoch in range(epochs):
    #     avg_loss = .0
    #     total_batch = int(mnist.train.num_examples/batch_size)
    #     for i in range(total_batch):
    #         batchx,batchy = mnist.train.next_batch(batch_size)
    #         _,l = sess.run([optimizer,loss],feed_dict={x:batchx,y:batchy,keep_prob:dropout})
    #         avg_loss += l/total_batch
    #     if epoch % display_step == 0:
    #         acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.})
    #         loss_list.append(avg_loss)
    #         acc_list.append(acc)
    #         val_list.append(sess.run(accuracy,feed_dict={x:mnist.validation.images,y:mnist.validation.labels,keep_prob:1.}))
    #         print('epoch:{0:d}, loss={1:.6f}, acc={2:.6f}'.format(epoch,avg_loss,acc))

    #write the data to summary
    summary_writer = tf.summary.FileWriter(path,graph=tf.get_default_graph())

    for i in range(1,num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
        if i % display_step ==0 or i==1:
            l,acc,summary = sess.run([loss,accuracy,merged_all],feed_dict={x:batch_x,y:batch_y,keep_prob:1.})
            print('step={0:d},loss={1:.7f},acc={2:.7f}'.format(i,l,acc))
            summary_writer.add_summary(summary,i)

    print('done!')

    test_data,test_label = mnist.test.images,mnist.test.labels
    print('test accuracy =',sess.run(accuracy,feed_dict={x:test_data,y:test_label,keep_prob:1.}))

    #plot
    fig = plt.figure(figsize=(8,6))
    plt.title('loss curve')
    plt.plot(loss_list, label='loss')
    fig2 = plt.figure(figsize=(8,6))
    plt.title('test and validation accuracy')
    plt.plot(acc_list, label='accuracy')
    plt.plot(val_list, label='validation')
    plt.show()
