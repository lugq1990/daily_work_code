# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

tf.enable_eager_execution()
tfe = tf.contrib.eager

class Model(object):
    def __init__(self):
        self.w = tfe.Variable(5.)
        self.b = tfe.Variable(0.)

    def __call__(self, x):
        return self.w*x + self.b

# loss function
def loss(pred, ytruth):
    return tf.reduce_mean(tf.square(pred-ytruth))

# training function
def train(model, inputs, outputs, lr=.1):
    with tfe.GradientTape() as t:
        curr_loss = loss(model(inputs), outputs)

    # compute the gradient
    dw, db = t.gradient(curr_loss, [model.w, model.b])
    model.w.assign_sub(lr*dw)
    model.b.assign_sub(lr*db)


# input data and output data
tw = 10.
tb = 3.
n = 1000
inputs = tf.random_normal([n])
noise = tf.random_normal([n])
outputs = inputs*tw + noise + tb

n_steps = 10
w_list, b_list = [], []

model = Model()
# start training loop
for i in range(n_steps):
    w_list.append(model.w.numpy())
    b_list.append(model.b.numpy())
    c_loss = loss(model(inputs), outputs)

    # start to train the model
    train(model, inputs, outputs, lr=.1)

    print('Step %d: w=%1.2f, b=%1.2f, loss=%2.4f'%(i, w_list[-1], b_list[-1], c_loss))


# plot the fitting result
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.scatter(inputs, outputs, c='r')
plt.scatter(inputs, model(inputs), c='b')
plt.legend()

# plot the training process w and b
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.plot(np.arange(n_steps), w_list, 'b', np.arange(n_steps), b_list, 'r')
plt.plot(np.arange(n_steps), [tw]*n_steps, 'b--', np.arange(n_steps), [tb]*n_steps, 'r--')
plt.legend()

plt.show()