# -*- coding:utf-8 -*-
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def propagate(w,b,x,y):
    m = x.shape[0]
    #x shape is 120*4, w shape is 4*1,y shape is 120*1
    pred = sigmoid(np.dot(w.T,x.T)+b)
    #pred shape is 120*1
    cost = -(np.dot(y,np.log(pred.T))+np.dot((1-y),np.log(1-pred.T)))/m
    print('cost ',cost)
    dw = np.dot(x.T,(pred-y).T)/m
    db = np.sum(pred-y)/m
    cost = np.squeeze(cost)
    grads = {'dw':dw,'db':db}
    return grads,cost

def backpropagage(w,b,x,y,steps=20000,learing_rate=.001):
    costs = []
    for i in range(steps):
        grads,cost = propagate(w,b,x,y)
        dw = grads['dw']
        db = grads['db']
        w = w - learing_rate*dw
        b = b - learing_rate*db
        if(i%500 ==0):
            print('step %d, cost = %f'%(i,cost))
            costs.append(cost)
    params = {'w':w,'b':b}
    grads = {'dw':dw,'db':db}
    return params,grads,costs

def predict(w,b,x):
    m = x.shape[0]
    prediction = np.zeros((1,m))
    w = w.reshape(x.shape[1],1)
    pred = sigmoid(np.dot(w.T,x.T)+b)
    for i in range(x.shape[0]):
        if(pred[0][i] <= .5):
            prediction[0][i] = 0
        else:
            prediction[0][i] = 1
    return prediction,pred

def init_param(dim):
    w = np.random.random((dim,1))
    b = 0
    return w,b


def model(xtrain,ytrain,xtest,ytest,iterations=200,lr=.01):
    w,b = init_param(xtrain.shape[1])
    params,grads,costs = backpropagage(w,b,xtrain,ytrain,steps=iterations,learing_rate=lr)
    print(params)
    w = params['w']
    b = params['b']
    pred_xtrain = predict(w,b,xtrain)
    pred_xtest =predict(w,b,xtest)
    print('train accuracy is %d'%(1-l2_loss(pred_xtrain,ytrain)))
    print('test accuracy is %d'%(1-l2_loss(pred_xtest,ytest)))

def l2_loss(pred,true):
    return np.mean(np.sum(np.square(pred-true)))

w,b,x,y = np.array([[1],[2]]),2,np.array([[2,2,2],[1,1,1]]),np.array([[1,0,1]])
# params,grads,costs = backpropagage(w,b,x,y)
# print(params)
# print(costs)
# print(grads)

#pred,prob = predict(params['w'],params['b'],x)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
x,y = load_iris(return_X_y=True)
x,y = shuffle(x,y)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=1234)
# ytrain = ytrain.reshape(-1,1)
# ytest = ytest.reshape(-1,1)

model(xtrain,ytrain,xtest,ytest)