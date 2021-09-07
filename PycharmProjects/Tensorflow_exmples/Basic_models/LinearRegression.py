# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = .01
epochs = 1000
display_step = 100
rng = np.random

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

x = tf.placeholder('float')
y = tf.placeholder('float')

w = tf.Variable(rng.randn(),name='weight')
b = tf.Variable(rng.randn(), name='bias')

pred = tf.add(tf.multiply(x,w),b)
cost = tf.reduce_sum(tf.pow(y-pred,2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_list = list()
    for epoch in range(epochs):
        for (x_i,y_i) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={x:x_i,y:y_i})

        #display
        if epoch % display_step ==0:
            loss = sess.run(cost,feed_dict={x:train_X,y:train_Y})
            loss_list.append(loss)
            print('epoch %d, loss= %f'%(epoch, loss))
            if(loss < loss_list[0]):
                color = plt.cm.get_cmap('nipy_spectral')(float(epochs)/n_samples)
                plt.plot(train_X, sess.run(w)*train_X+sess.run(b), color=color)

    print('done!')
    train_cost = sess.run(cost,feed_dict={x:train_X,y:train_Y})
    print('train loss=',train_cost,'w=',sess.run(w),'b=',sess.run(b))

    #plot the data
    plt.plot(train_X, train_Y, 'ro', label='original data')
    plt.plot(train_X, sess.run(w)*train_X+sess.run(b), label='fitted')
    plt.legend()
    plt.show()

