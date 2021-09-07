# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

learning_rate = .001
batch_size = 128
num_steps = 2000

num_input = 784
num_classes = 10
dropout = .25


#create the network structure
def conv_net(x,n_classes,dropout,reuse,is_training):
    with tf.variable_scope('conv_net',reuse=reuse):
        x = x['images']
        x = tf.reshape(x,shape=[-1,28,28,1])
        conv1 = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1,2,2)
        conv2 = tf.layers.conv2d(conv1,64,5,activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2,2,2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1,1024)
        fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)
        out = tf.layers.dense(fc1,n_classes)
    return out

#make the estimator using the TF estimator template
def model_fn(features,labels,mode):
    #for the dropout,it is different for training and test
    logits_train = conv_net(features,n_classes=num_classes,dropout=dropout,reuse=False,is_training=True)
    logits_test = conv_net(features,n_classes=num_classes,dropout=dropout,reuse=True,is_training=False)
    pred_classes = tf.argmax(logits_test,axis=1)
    pred_probs = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_classes)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train,labels=tf.cast(labels,tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)
    estimator = tf.estimator.EstimatorSpec(mode=mode,predictions=pred_classes,
                                           loss=loss_op,train_op=train_op,
                                           eval_metric_ops={'accuracy':acc_op})
    return estimator

model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images},
                                              y=mnist.train.labels,
                                              batch_size=batch_size,
                                              num_epochs=None,
                                              shuffle=True)
print('start train the model')
#train the model
model.train(input_fn=input_fn,steps=num_steps)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},
                                                    y = mnist.test.labels,
                                                   batch_size=batch_size,
                                                   shuffle=False)
res = model.evaluate(test_input_fn,steps=1)
print('test accuracy :',res['accuracy'])

