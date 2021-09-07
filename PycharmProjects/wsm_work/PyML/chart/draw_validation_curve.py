# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 09:30:39 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.learning_curve import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression


def DRAW(X, y, param_name, param_range, scl=None, pca=None, clf=LogisticRegression(), cv=10, ylim=[0.1, 1]):
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
        pipe_lr = Pipeline([('clf', clf)])
    
    train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X, y=y,
                                                        param_name="clf__%s" %param_name, param_range=param_range,
                                                        cv=10)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.semilogx(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.2, color='blue')
    plt.semilogx(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.2, color='green')
    
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter %s' %param_name)
    plt.ylabel('Accuracy')
    plt.ylim(ylim)
    plt.show()

