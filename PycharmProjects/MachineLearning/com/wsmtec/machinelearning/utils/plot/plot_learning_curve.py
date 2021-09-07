# -*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(clf,x,y,cv=5,shuffle=True,figsize=None,title='learning curve',textsize='medium'):
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.set_title(label=title,fontsize=textsize)
    ax.set_xlabel('training examples',fontsize=textsize)
    ax.set_ylabel('score',fontsize=textsize)
    train_sizes,train_score,test_score = learning_curve(clf,x,y,cv=cv,shuffle=shuffle)
    train_mean = np.mean(train_score,axis=1)
    train_std = np.std(train_score,axis=1)
    test_mean = np.mean(test_score,axis=1)
    test_std = np.std(test_score,axis=1)
    ax.grid()
    ax.fill_between(train_sizes,train_mean-train_std,train_mean+train_std,alpha=.1,color='r')
    ax.fill_between(train_sizes,test_mean-test_std,test_mean+test_std,alpha=.1,color='g')
    ax.plot(train_sizes,train_mean,'o-',color='r',label='training score')
    ax.plot(train_sizes,test_mean,'o-',color='g',label='validation score')
    ax.tick_params(labelsize=textsize)
    ax.legend(loc='best',fontsize=textsize)

    return ax

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.datasets import load_digits
x,y = load_digits(return_X_y=True)
from sklearn.utils import shuffle
x,y = shuffle(x,y)
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=3,test_size=.2,random_state=1234)

plot_learning_curve(lr,x,y,cv=cv)
plt.show()