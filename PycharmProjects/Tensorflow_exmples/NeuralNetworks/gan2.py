# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

learning_rate = .0002
batch_size = 128
num_steps = 30000
display_step = 1000

n_input = 784
gen_units = 512
disc_units = 512
noise_dim = 100

#glorot init
def glorot_init(shape):
    return tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2.0))

#weights and biases
weights = {'gen_hidden':tf.Variable(glorot_init([noise_dim,gen_units])),
           'gen_out':tf.Variable(glorot_init([gen_units,n_input])),
           'disc_hidden':tf.Variable(glorot_init([n_input,disc_units])),
           'disc_out':tf.Variable(glorot_init([disc_units,1]))}
biases = {'gen_hidden':tf.Variable(glorot_init([gen_units])),
          'gen_out':tf.Variable(glorot_init([n_input])),
          'disc_hidden':tf.Variable(glorot_init([disc_units])),
          'disc_out':tf.Variable(glorot_init([1]))}

#define the generator and desciminator
def generator(x):
    hidden = tf.add(tf.matmul(x,weights['gen_hidden']),biases['gen_hidden'])
    hidden = tf.nn.relu(hidden)
    out = tf.add(tf.matmul(hidden,weights['gen_out']),biases['gen_out'])
    out = tf.nn.sigmoid(out)
    return out

def discriminator(x):
    hidden = tf.add(tf.matmul(x,weights['disc_hidden']),biases['disc_out'])
    hidden = tf.nn.relu(hidden)
    out = tf.add(tf.matmul(hidden,weights['disc_out']),biases['disc_out'])
    out = tf.nn.sigmoid(out)
    return out

#define the gen and disc inputs
gen_inputs = tf.placeholder(tf.float32,shape=[None, noise_dim],name='gen_input')
disc_inputs = tf.placeholder(tf.float32,shape=[None,n_input],name='disc_input')

#get the gen samples
gen_samples = generator(gen_inputs)
#get the real and fake disc samples
disc_fake = discriminator(gen_samples)
disc_real = discriminator(disc_inputs)

#define the loss function
#the gen
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real)+tf.log(1-disc_fake))

#define the optimizer
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

#get the gen and disc vars
gen_vars = [weights['gen_hidden'],weights['gen_out'],biases['gen_hidden'],biases['gen_out']]
disc_vars = [weights['disc_hidden'],weights['disc_out'],biases['disc_hidden'],biases['disc_out']]

#define the train_op for gen and disc
train_gen = optimizer_gen.minimize(gen_loss,var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

#init all variables
init = tf.global_variables_initializer()

#start the session
sess = tf.Session()

sess.run(init)

#start the for loop
gen_loss_list = list()
disc_loss_list = list()
for i in range(num_steps):
    batchx, _ = mnist.train.next_batch(batch_size)
    z = np.random.uniform(-1.,1.,size=[batch_size, noise_dim])

    #start to train the model
    _,_,gen_l,disc_l = sess.run([train_disc,train_gen,gen_loss,disc_loss],feed_dict={gen_inputs:z,disc_inputs:batchx})
    gen_loss_list.append(gen_l)
    disc_loss_list.append(disc_l)

    if i % display_step == 0:
        print('step={0:d},gen loss={1:.7f},disc loss={2:.7f}'.format(i,gen_l,disc_l))

#plot the gen and disc loss
plt.plot(gen_loss_list)
plt.plot(disc_loss_list)
plt.title('generator and discriminator loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.show()

n = 6
canvas = np.empty((28 * n, 28 * n))
for j in range(n):
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # get the generative samples
    g = sess.run(gen_samples, feed_dict={gen_inputs: z})
    g = -1 * (g - 1)
    for t in range(n):
        canvas[j * 28:(j + 1) * 28, t * 28:(t + 1) * 28] = g[t].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin='upper', cmap='gray')
plt.show()