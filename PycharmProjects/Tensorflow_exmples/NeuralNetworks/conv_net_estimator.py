# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

#hyperparameters
learning_rate = .001
batch_size = 128
num_steps = 2000
display_step = 100
n_input = 784
n_classes = 10
dropout = .5

# define the neural net
def conv_net(x_dict,n_classes,dropout,reuse,istraining):
    with tf.variable_scope('ConvNet',reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x,[-1,28,28,1])
        conv1 = tf.layers.conv2d(x,32,5,padding='SAME')
        conv1 = tf.layers.max_pooling2d(conv1,2,2,padding='SAME')
        conv2 = tf.layers.conv2d(conv1,64,5,padding='SAME')
        conv2 = tf.layers.max_pooling2d(conv2,2,2,padding='SAME')
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1,1024)
        fc1 = tf.layers.dropout(fc1,rate=dropout,training=istraining)
        fc1 = tf.layers.dense(fc1,n_classes)
        return fc1

#define the model_fn
def model_fn(features,labels,mode):
    logits_train = conv_net(features,n_classes,dropout,reuse=False,istraining=True)
    logits_test = conv_net(features,n_classes,dropout,reuse=True,istraining=False)

    pred_class = tf.argmax(logits_test,axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_class)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_test,
                                                                            labels=tf.cast(labels,tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_class)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss_op,
                                      train_op=train_op,
                                      predictions=pred_class,
                                      eval_metric_ops={'accuracy':acc_op})

#define the input_fn
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images},
                                              y=mnist.train.labels,
                                              batch_size=batch_size,shuffle=True,num_epochs=num_steps)

#get the model by tf.estimator.Estimator
model = tf.estimator.Estimator(model_fn)

#start to train the model
model.train(input_fn,steps=num_steps)

#evaluate the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},
                                                   y=mnist.test.labels,
                                                   shuffle=False,num_epochs=1)
result = model.evaluate(eval_input_fn,steps=1)
print('result is ',result)