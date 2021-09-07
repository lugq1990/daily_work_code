# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

learning_rate = .01
batch_size = 128
num_step = 1000
display_step = 1

hidden_1 = 256
hidden_2 = 256
n_classes = 10

#define the input function
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images},y=mnist.train.labels,
                                              batch_size=batch_size,num_epochs=num_step,shuffle=True)

#define the neural net
def neural_net(x_dict):
    x = x_dict['images']
    layer1 = tf.layers.dense(x, hidden_1)
    layer2 = tf.layers.dense(layer1, hidden_2)
    return tf.layers.dense(layer2,n_classes)

#define the estimator function
def model_fn(features,labels,mode):
    logits = neural_net(features)
    pred = tf.argmax(logits,axis=1)

    #if the predict, return first
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.cast(labels,tf.int64)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(predictions=pred,labels=labels)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=pred,
                                      loss=loss, train_op=train_op,
                                      eval_metric_ops={'accuracy':acc_op})

model = tf.estimator.Estimator(model_fn)

#start to train the model
model.train(input_fn, steps=num_step)

#evaluate the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},y=mnist.test.labels,
                                                   num_epochs=1,shuffle=False)
result = model.evaluate(eval_input_fn)
print('model result is ',result)

#get the test images
num = 5
test_images = mnist.test.images[:num]
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':test_images},num_epochs=1,shuffle=False)

preds = list(model.predict(test_input_fn))
labels = list(mnist.test.labels[:num])

fig = plt.figure(figsize=(10, 8))
for i in range(num):
    plt.subplot(1, num, i+1)
    plt.imshow(np.reshape(test_images[i],[28,28]),cmap='gray')
    plt.title('True:'+np.str(labels[i])+' Pred:'+np.str(preds[i]))

plt.show()