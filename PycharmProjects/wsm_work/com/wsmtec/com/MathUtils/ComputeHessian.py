# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

#define the constant data placeholder
def cons(x):
    return tf.constant(x,dtype=tf.float32)

def compute_hessian(fn,vars):
    mat = []
    for var1 in vars:
        tmp = []
        for var2 in vars:
            tmp.append(tf.gradients(tf.gradients(fn,var2)[0],var1)[0])
        #incase the gradient is none
        tmp = [cons(0) if t==None else t for t in tmp]
        tmp = tf.stack(tmp)
        mat.append(tmp)
    mat = tf.stack(mat)
    return mat

x = tf.Variable(np.random.random_sample(),dtype=tf.float32)
y = tf.Variable(np.random.random_sample(),dtype=tf.float32)

#for example for fun = x**2 + y**2 +2*x*y
fn = tf.pow(x,cons(2)) + tf.pow(y,cons(2) + 2*x*y)

hessian = compute_hessian(fn,[x,y])
#init the variable
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#compute the hessian for delivarate of x and y
re = sess.run(hessian)
print(re)