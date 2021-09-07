# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

learning_rate = .01
batch_size = 128
num_steps = 1000
display_step = 1

hidden_1 = 256
hidden_2 = 256

#define the input function
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images},
                                              y=mnist.train.labels,
                                              batch_size=batch_size,num_epochs=None, shuffle=True)

#define the neural network
def neural_net(x_dict):
    x = x_dict['images']
    layer1 = tf.layers.dense(x, hidden_1)
    layer2 = tf.layers.dense(layer1, hidden_2)
    out = tf.layers.dense(layer2, 10)
    return out

#Define the model function
def model_fn(features,labels,mode):
    logits = neural_net(features)

#    pred = tf.nn.softmax(logits)
    pred_classes = tf.argmax(logits, axis=1)

    #Define the preditction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    #Define the loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.cast(labels,tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #evaluate the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes,
                                      loss=loss,train_op=train_op,
                                      eval_metric_ops={'accuracy':acc_op})

model = tf.estimator.Estimator(model_fn)

#start train the model
model.train(input_fn, steps=num_steps)

#evaluate  the model
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},y=mnist.test.labels,
                                                   num_epochs=1,shuffle=False)
result = model.evaluate(input_fn=test_input_fn,steps=1)
print(result)

#plot the true-prediction result
test_images = mnist.test.images[:4]
test_fn = tf.estimator.inputs.numpy_input_fn(x={'images':test_images},shuffle=False)

preds = list(model.predict(test_input_fn))
trues = list(mnist.test.labels[:4])
for i in range(4):
    plt.imshow(np.reshape(test_images[i],[28,28]),cmap='gray')
    plt.title('true is '+np.str(trues[i])+' and predicted is '+np.str(preds[i]))
    plt.show()
    #print('prediction is ',np.str(preds[i]))
