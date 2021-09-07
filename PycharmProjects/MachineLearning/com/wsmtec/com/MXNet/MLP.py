# -*- coding:utf-8 -*-
import numpy as np
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.ERROR)

mnist = mx.test_utils.get_mnist()
batch_size =100
train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size,shuffle=True)
test_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size)

data = mx.sym.var("data")
