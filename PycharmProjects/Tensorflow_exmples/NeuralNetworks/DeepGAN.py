# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

num_steps = 10000
lr_gen = .002
lr_disc = .002
batch_size = 128

image_dim = 784
noise_dim = 100

#build the placeholder for gen and disc
noise_input = tf.placeholder(tf.float32,[None,noise_dim])
image_input = tf.placeholder(tf.float32,[None,28,28,1])
is_training = tf.placeholder(tf.bool)

#define the leaky relu
def leakyrelu(x,alpha=.2):
    return .5*(1+alpha)*x + .5*(1-alpha)*tf.abs(x)

#define the gen
def generator(x, reuse=False):
    with tf.variable_scope('Generator',reuse=reuse):
        x = tf.layers.dense(x,units=7*7*128)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

        x = tf.reshape(x,shape=[-1,7,7,128])
        x = tf.layers.conv2d_transpose(x,64,5,strides=2,padding='same')
        x = tf.layers.batch_normalization(x,training=is_training)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x,1,5,strides=2,padding='same')
        x = tf.nn.tanh(x)
        return x

#define the disc
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator',reuse=reuse):
        x = tf.layers.conv2d(x,64,5,strides=2,padding='same')
        x = tf.layers.batch_normalization(x,training=is_training)
        x = leakyrelu(x)

        x = tf.layers.conv2d(x,32,5,strides=2,padding='same')
        x = tf.layers.batch_normalization(x,training=is_training)
        x = leakyrelu(x)

        x = tf.reshape(x,shape=[-1,7*7*128])
        x = tf.layers.dense(x,units=1024)
        x = tf.layers.batch_normalization(x,training=is_training)
        x = leakyrelu(x)

        x = tf.layers.dense(x,2)
    return x

#build the gen net
gen_samples = generator(noise_input)
#build the disc net
disc_real = discriminator(image_input)
disc_fake = discriminator(gen_samples,reuse=True)

#build the stacked disc
stacked_gan = discriminator(gen_samples,reuse=True)

#disc loss
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real,
                                                                               labels=tf.ones([batch_size],dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake,
                                                                               labels=tf.zeros([batch_size],dtype=tf.int32)))
#disc loss
disc_loss = disc_real+disc_fake
#gen loss
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=stacked_gan,
                                                                         labels=tf.ones([batch_size],dtype=tf.int32)))

#build the optimizer
optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_gen,beta1=.5,beta2=.999)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_disc,beta1=.5,beta2=.999)

#precise the gen and disc vars
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')

train_gen = optimizer_gen.minimize(gen_loss,var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss,var_list=disc_vars)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

gen_loss_list = list()
disc_loss_list = list()
for i in range(num_steps):
    batchx,_ = mnist.train.next_batch(batch_size)
    batchx = np.reshape(batchx,[-1,28,28,1])
    batchx = batchx*2.-1.

    #generator noise images
    z = np.random.uniform(-1,1.,size=[batch_size,noise_dim])
    _,_,gl,dl = sess.run([train_gen,train_disc,gen_loss,disc_loss],feed_dict={noise_input:z,image_input:batchx,is_training: True})

    gen_loss_list.append(gl)
    disc_loss_list.append(dl)
    if i % 500 ==0 or i ==0:
        print('step {0:d},gen loss={1:.7f},disc loss={2:.7f}'.format(i,gl,dl))

#plot the loss curve
plt.plot(gen_loss_list)
plt.plot(disc_loss_list)
plt.title('gen and disc loss')
plt.ylabel('loss')
plt.xlabel('steps')


n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # Generate image from noise.
    g = sess.run(gen_samples, feed_dict={noise_input: z, is_training:False})
    # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
    g = (g + 1.) / 2.
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()




