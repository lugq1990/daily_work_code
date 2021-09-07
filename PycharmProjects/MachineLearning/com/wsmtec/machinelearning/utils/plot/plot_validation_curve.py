# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

def plot_validation_curve(clf,x,y,cv=None,cv_split=5,test_size=.2,random_state=None,
                          param_name=None,param_range=None,
                          scoring=None,textsize='large',lw=2,
                          ax=None,figsize=None,return_best_param=False):
    if(ax is None):
        fig,ax = plt.subplots(1,1,figsize=figsize)
    if(cv is None):
        cv = ShuffleSplit(n_splits=cv_split,test_size=test_size,random_state=random_state)
    if(param_range is None):
        param_range = np.logspace(-6,-1,5)

    train_scores,test_scores = validation_curve(clf,x,y,cv=cv,param_name=param_name,param_range=param_range,
                                                scoring=scoring)
    train_score_mean = np.mean(train_scores,axis=1)
    train_score_std = np.std(train_scores,axis=1)
    test_score_mean = np.mean(test_scores,axis=1)
    test_score_std = np.std(test_scores,axis=1)
    best_param_index = test_score_mean.argmax()
    best_param = param_range[best_param_index]

    ax.set_title('valiation curve',fontsize=textsize)
    ax.set_xlabel(param_name,fontsize=textsize)
    ax.set_ylabel('score',fontsize=textsize)
    plt.ylim(0.0,1.1)
    plt.axvline(best_param,linestyle=':',label='best parameter')
    plt.grid()
    plt.plot(param_range,train_score_mean,label='training score',color='r',lw=lw)
    # plt.semilogx(param_range,train_score_mean,label='training score',color='darkorange',lw=lw)
    plt.fill_between(param_range,train_score_mean-train_score_std,train_score_mean+train_score_std,
                     color='r',lw=lw,alpha=.2)
    # plt.semilogx(param_range,test_score_mean,label='cross-validation score',color='navy',lw=lw)
    plt.plot(param_range,test_score_mean,label='validation score',color='g',lw=lw)
    plt.fill_between(param_range,test_score_mean-test_score_std,test_score_mean+test_score_std,
                     color='g',lw=lw,alpha=.2)
    plt.legend(loc='best')
    if(return_best_param):
        return best_param,ax
    else:
        return ax

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
x,y = load_digits(return_X_y=True)
x,y = shuffle(x,y)
clf = SVC()
a,_=plot_validation_curve(clf,x,y,scoring='accuracy',param_name='C',param_range=[.1,10,100],return_best_param=True)
plt.show()


