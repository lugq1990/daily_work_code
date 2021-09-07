# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 09:30:39 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression


def DRAW(X, y, scl=None, pca=None, clf=LogisticRegression(),
                        train_sizes=np.linspace(.1, 1., 10), cv=10, ylim=[0.1, 1]):
    if scl and pca:
        pipe_lr = Pipeline([('scl', scl), 
                            ('pca', pca), 
                            ('clf', clf)])
    elif scl:
        pipe_lr = Pipeline([('scl', scl), 
                            ('clf', clf)])
    elif pca:
        pipe_lr = Pipeline([('pca', pca), 
                            ('clf', clf)])
    else:
        pipe_lr = clf
    
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X, y=y,
                                                            train_sizes=train_sizes, cv=cv, n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim(ylim)
    plt.show()



