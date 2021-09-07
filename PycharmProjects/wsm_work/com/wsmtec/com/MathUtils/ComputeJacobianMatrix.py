# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

def cons(x):
    return tf.constant(x,dtype=tf.float32)

def compute_jacobian(fns,vars):
    mat = []
    for fn in fns:
        tmp = []
        for var in vars:
            tmp.append(tf.gradients(fn,var)[0])
        tmp = [0 if t==None else t for t in tmp]
        tmp = tf.stack(tmp)
        mat.append(tmp)
    mat = tf.stack(mat)
    return mat

x = tf.Variable(np.random.random_sample(),dtype=tf.float32)
y = tf.Variable(np.random.random_sample(),dtype=tf.float32)

fn1 = tf.pow(x,cons(2)) + x + y
fn2 = tf.pow(x,cons(2)) + tf.pow(y,cons(2)) + x*y
fn = [fn1,fn2]

jacobian = compute_jacobian(fn,[x,y])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

re = sess.run(jacobian)
print(re)

