# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define the hyperparameters
learning_rate = .001
num_steps = 10000
batch_size = 64
input_dim = 784
hidden_dim = 512
latent_dim = 2

# define the input placeholder for the image
input_image = tf.placeholder(tf.float32, [None, input_dim])

# define the Xavier initialization
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1./(tf.sqrt(shape[0]/2.)))

# define all the variables, for weights and bias
weights = {'encoder_h1': tf.Variable(glorot_init([input_dim, hidden_dim])),
           'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
           'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
           'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
           'decoder_out': tf.Variable(glorot_init([hidden_dim, input_dim]))}
biases = {'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
          'z_mean': tf.Variable(glorot_init([latent_dim])),
          'z_std': tf.Variable(glorot_init([latent_dim])),
          'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
          'decoder_out': tf.Variable(glorot_init([input_dim]))}

# building the encoder
encoder = tf.nn.tanh(tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1'])
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# get the gaussian random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1., name='epsilon')
z = z_mean + tf.exp(z_std/ 2)*eps

# building the decoder
decoder = tf.nn.tanh(tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1'])
decoder = tf.nn.sigmoid(tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out'])

# define the VAE loss
def vae_loss(x_reconstructed, x_true):
    encoder_decoder_loss = x_true*tf.log(1e-10 + x_reconstructed)+ (1-x_true)*tf.log(1e-10+1-x_reconstructed)
    encoder_decoder_loss = - tf.reduce_sum(encoder_decoder_loss, 1)
    # KL divergence loss
    kl_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_loss = -.5*tf.reduce_sum(kl_loss, 1)
    return tf.reduce_mean(encoder_decoder_loss + kl_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss_op)

init = tf.global_variables_initializer()

# start to train the model
sess = tf.Session()
sess.run(init)
for i in range(num_steps + 1):
    batch_x, _ = mnist.train.next_batch(batch_size)
    _, l = sess.run([optimizer, loss_op], feed_dict={input_image: batch_x})
    if i % 1000 == 0 :
        print('Now is steps %i, Loss = %f'%(i, l))


# start to generate images, using decoder function
noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
# Building a manifold of generated digits
n = 20
x_axis = np.linspace(-3, 3, n)
y_axis = np.linspace(-3, 3, n)
canvas = np.empty((28 * n, 28 * n))
for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        z_mu = np.array([[xi, yi]] * batch_size)
        x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
        x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_axis, y_axis)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()


from tensorflow.contrib import rnn
a = rnn.BasicLSTMCell()
outputs,_,_ = rnn.static_bidirectional_rnn
