# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

learning_rate = .001
batch_size = 128
steps = 10000
display_step = 1000

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_input], name='input_data')
y = tf.placeholder(tf.float32,[None,n_classes], name='label')

#define the weights and biases
weights = {'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1]),name='w1'),
           'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name='w2'),
           }




