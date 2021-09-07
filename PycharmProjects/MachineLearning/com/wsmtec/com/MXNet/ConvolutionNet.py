# -*- coding:utf-8 -*-
import mxnet as mx
import logging
#logging.getLogger().setLevel(logging.DEBUG)

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size,shuffle=True)
test_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size,shuffle=False)
data = mx.sym.var('data')
conv1 = mx.sym.Convolution(data=data,kernel=(5,5),num_filter=20)
relu1 = mx.sym.Activation(data=conv1,act_type='relu')
pool1 = mx.sym.Pooling(data=relu1,pool_type='max',kernel=(2,2),stride=(2,2))
conv2 = mx.sym.Convolution(data=pool1,kernel=(5,5),num_filter=50)
relu2 = mx.sym.Activation(data=conv2,act_type='relu')
