# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import load_iris

style.use('fivethirtyeight')

iris = load_iris()
x, y = iris.data, iris.target
x, y = x[:100, :], y[:100]

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def stable_softmax(x):
    return np.exp(x - max(x)) / np.sum(np.exp(x - max(x)))

def init_w_b(shape):
    w = np.random.normal(loc=0, scale=1, size=shape[1])
    b = np.random.normal(loc=0, scale=1, size=1)
    return w, b

# compute the cross entropy
def cross_entropy(y, pred):
    return -sum(y*np.log(pred) + (1-y)*np.log(1-pred))

w, b = init_w_b(x.shape)

def logist(x, y, iter=10):

    for i in range(iter):
        # forward propagation
        logits = sigmoid(np.dot(x, w) + b)
        pred = np.argmax(logits, axis=0)
        loss = cross_entropy(y, pred)

