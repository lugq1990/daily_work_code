# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
x,y = load_iris(return_X_y=True)
x,y = shuffle(x,y)

#hyper
learning_rate = .01
n_input = x.shape[1]
n_output = np.unique(y).shape[0]
n_hidden = 4

def init(n_input,n_output,n_hidden):
    w1 = np.random.uniform(-1.,1.,size=[n_input,n_hidden])
    b1 = np.zeros([n_hidden])
    w2 = np.random.uniform(-1.,1.,size=[n_hidden,n_output])
    b2 = np.zeros([n_output])
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    return params

def forward_propagate(x,params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    z1 = np.dot(x,w1)+b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1,w2)+b2
    a2 = np.tanh(z2)
    cache = {'z1':z1,'a1':a1,'z2':z2,'a2':a2}
    return a2,cache

#compute the loss
def compute_cost(a2,y):
    m = y.shape[0]
    print('a2 shape',a2.shape)
#    cost = -(np.dot(y,np.log(a2.T))+np.dot(np.log(1-a2.T),(1-y).T))/m
    cost = np.multiply(np.log(a2),y) + np.multiply(np.log(1-a2),(1-y))
    cost = -np.sum(cost)/m
    cost = np.squeeze(cost)
    return cost

#back propagate
def back_proipagate(x,y,params,cache):
    m = x.shape[0]

    w1 = params['w1']
    w2 = params['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - y
    dw2 = np.dot(dz2,a1.T)/m
    db2 = np.sum(dz2,axis=1,keepdims=True)/m
    dz1 = np.dot(w2.T,dw2)*(1-np.power(a1,2))
    dw1 = np.dot(dz1,x.T)/m
    db1 = np.sum(dz1,axis=1,keepdims=True)/m

    grads = {'dw1':dw1,'db1':db1,'dw2':dw2,'db2':db2}
    return grads

def update_parameters(params,grads,learning_rate=learning_rate):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - learning_rate*dw1
    b1 = b1 -learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2

    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    return params

def model(x,y,n_input=n_input,n_hidden=n_hidden,n_output=n_output,steps=10000,print_cost=True):
    params = init(n_input,n_output,n_hidden)
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    for i in range(steps):
        a2,cache = forward_propagate(x,params)
        cost = compute_cost(a2,y)
        grads = back_proipagate(x,y,params,cache)
        params = update_parameters(params,grads)

        if i % 200==0:
            print('step={0:d},loss={1:.7f}'.format(i,cost))

    return params

def predict(params,x):
    a2,cache = forward_propagate(x,params)
    predictions = np.round(a2)
    return predictions

y = y.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(y)
print('y shape',y.shape)
model(x,y)
