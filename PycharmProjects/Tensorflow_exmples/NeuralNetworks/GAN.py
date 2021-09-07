# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

start_time = time.time()
path = 'E:\\tensorboard\\20180301'
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparameters
learning_rate = .0002
batch_size = 128
num_steps = 10000

#network parameters
image_dim = 784
gen_units = 256
disc_units = 256
noise_dim = 100

#use the glorot init
def glorot_init(shape):
    return tf.random_normal(shape=shape,stddev=1./(tf.sqrt(shape[0]/2.0)))

#define the weights and biases
weights = {'gen_hidden':tf.Variable(glorot_init([noise_dim,gen_units])),
           'gen_out':tf.Variable(glorot_init([gen_units,image_dim])),
           'disc_hidden':tf.Variable(glorot_init([image_dim,disc_units])),
           'disc_out':tf.Variable(glorot_init([disc_units,1]))}
biases = {'gen_hidden':tf.Variable(glorot_init([gen_units])),
          'gen_out':tf.Variable(glorot_init([image_dim])),
          'disc_hidden':tf.Variable(glorot_init([disc_units])),
          'disc_out':tf.Variable(glorot_init([1]))}

#define the generative and discriminator function
def generator(x):
    hidden_layer = tf.add(tf.matmul(x,weights['gen_hidden']),biases['gen_hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.add(tf.matmul(hidden_layer,weights['gen_out']),biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def discriminator(x):
    hidden_layer = tf.add(tf.matmul(x,weights['disc_hidden']),biases['disc_hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.add(tf.matmul(hidden_layer,weights['disc_out']),biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

#define the network inputs
gen_input = tf.placeholder(tf.float32,shape=[None,noise_dim],name='noise_input')
disc_input = tf.placeholder(tf.float32,shape=[None,image_dim],name='disc_input')

#build the generative net
with tf.name_scope('gen_samples'):
    gen_sample = generator(gen_input)

#build 2 discriminator net, one from the training data and the other from the generative net produced
with tf.name_scope('disc_real'):
    disc_real = discriminator(disc_input)
with tf.name_scope('disc_fake'):
    disc_fake = discriminator(gen_sample)

#build the loss, one for generative and one for discriminator loss(use the log loss)
with tf.name_scope('gen_loss'):
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))    #because the samples generatived use the discriminator to judge whether is sampled

with tf.name_scope('disc_loss'):                                                #from the data or from the fake samples
    disc_loss = - tf.reduce_mean(tf.log(disc_real)+tf.log(1-disc_fake))

#build 2 optimizer
with tf.name_scope('gen_optimizer'):
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
with tf.name_scope('disc_optimizer'):
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

#because there is 2 nets, each net must be optimizer separately. So the variables must be choosen explicitly
#the generative vars
gen_vars = [weights['gen_hidden'],weights['gen_out'],biases['gen_hidden'],biases['gen_out']]
#the discriminator vars
disc_vars = [weights['disc_hidden'],weights['disc_out'],biases['disc_hidden'],biases['disc_out']]

#create the training operations
with tf.name_scope('train_gen'):
    train_gen = optimizer_gen.minimize(gen_loss,var_list=gen_vars)
with tf.name_scope('train_disc'):
    train_disc = optimizer_disc.minimize(disc_loss,var_list=disc_vars)

#init all varaibles
init = tf.global_variables_initializer()

tf.summary.scalar('gen_loss',gen_loss)
tf.summary.scalar('disc_loss',disc_loss)

merge_all_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
    gen_loss_list = list()
    disc_loss_list = list()
    for i in range(num_steps):
        batchx,_ = mnist.train.next_batch(batch_size)
        #define the noise input for the generative inputs
        z = np.random.uniform(-1.,1.,size=[batch_size,noise_dim])

        #train the model
        _,_,gl,dl,summary = sess.run([train_disc,train_gen,gen_loss,disc_loss,merge_all_op],
                                     feed_dict={disc_input:batchx,gen_input:z})

        summary_writer.add_summary(summary,i)
        gen_loss_list.append(gl)
        disc_loss_list.append(dl)
        if i % 1000 == 0 or i ==0 :
            print('step={0:d},generator loss={1:.7f},discriminator loss={2:.7f}'.format(i,gl,dl))

    print('done!')

    #plot the generator loss and discriminator loss
    fig = plt.figure(figsize=(10,8))
    plt.plot(gen_loss_list,label='generative loss')
    plt.plot(disc_loss_list,label='discriminator loss')
    plt.title('generative and discriminator loss')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()


    #plot the generative samples
    n = 6
    canvas = np.empty((28*n,28*n))
    for j in range(n):
        z = np.random.uniform(-1.,1.,size=[n,noise_dim])
        #get the generative samples
        g = sess.run(gen_sample,feed_dict={gen_input:z})
        g = -1*(g-1)
        for t in range(n):
            canvas[j*28:(j+1)*28,t*28:(t+1)*28] = g[t].reshape([28,28])

    plt.figure(figsize=(n,n))
    plt.imshow(canvas,origin='upper',cmap='gray')
    plt.show()

print('all proceture use '+np.str((time.time()-start_time)/60)+" minutes")
