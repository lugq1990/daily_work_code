# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
path = 'E:\\tensorboard\\20180306'

#hyperparameters
learning_rate = .01
steps = 1000
batch_size = 256
display_step = 100

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256

#placeholder for image
x = tf.placeholder(tf.float32,[None,n_input],name='model_input')
y = tf.placeholder(tf.float32,[None,n_input],name='data')

#the weights and biases
weights = {'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1]),name='encoder_h1'),
           'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name='encoder_h2'),
           'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1]),name='decoder_h1'),
           'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]),name='decoder_h2')}

biases = {'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1]),name='encoder_b1'),
          'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2]),name='encoder_b2'),
          'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1]),name='decoder_b1'),
          'decoder_b2':tf.Variable(tf.random_normal([n_input]),name='decoder_b2')}

#build the encoder net
def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_h2']),biases['encoder_b2']))
    return layer2

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decoder_h2']),biases['decoder_b2']))
    return layer2

#build the model
with tf.name_scope('encoder_op'):
    encoder_op = encoder(x)
with tf.name_scope('decoder_op'):
    decoder_op = decoder(encoder_op)

#pred
with tf.name_scope('pred'):
    pred = decoder_op
#labels

#define the loss
with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.pow(y-pred, 2))
    loss = tf.losses.mean_squared_error(y, pred)
#define the optimizer
with tf.name_scope('optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
#summary
tf.summary.scalar('loss',loss)

merge_all_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(init)

loss_list = list()
for i in range(steps):
    summary_writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())

    batchx,_ = mnist.train.next_batch(batch_size)
    x_input = batchx + np.random.uniform(-.1,.1,size=[batch_size,n_input])
    _,l,summary = sess.run([optimizer,loss,merge_all_op],feed_dict={x:x_input, y:batchx})

    loss_list.append(l)

    if i % display_step ==0 :
        print('step:{0:d},loss={1:.7f}'.format(i,l))
        summary_writer.add_summary(summary, i)

#plot the loss
plt.plot(loss_list)
plt.title('loss curve')
plt.show()


# Testing
# Encode and decode images from test set and visualize their reconstruction.
n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    batch = batch_x + np.random.uniform(-.1,1,size=[n,n_input])
    # Encode and decode the digit image
    g = sess.run(decoder_op, feed_dict={x: batch, y:batch_x})

    # Display original images
    for j in range(n):
        # Draw the generated digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw the generated digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.title('original image')
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.title('new image')
plt.show()




