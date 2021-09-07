# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:03:00 2017

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class SVD(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_components=5):
        self.orig_X = 0
        self.U = 0
        self.partial_U = 0
        self.gamma = 0
        self.n_components = n_components
        
    def fit(self, X):
        self.orig_X = X
        self.U, self.gamma, V = np.linalg.svd(X)
        self.partial_U = self.U[:, :self.n_components]
        
    def transform(self, X):
        knn = KNeighborsClassifier(1)
        knn.fit(self.orig_X, np.arange(self.orig_X.shape[0]))
        indices = knn.predict(X)
        return self.partial_U[indices, :]
        
    def fit_transform(self, X):
        self.fit(X)
        return self.partial_U
        
    def plot(self):
        plt.plot(range(1, len(self.gamma)+1), self.gamma, marker='o')
        plt.xlabel('The number of gamma')
        plt.ylabel('Score')
        plt.show()
    

if __name__ == "__main__":
    from sklearn.datasets import base
    iris = base.load_iris()
    X0 = iris.data[11:,:]
    X1 = iris.data[:10,:]
    svd = SVD(3)
    svd.fit(X0)
    svd.plot()
    print svd.transform(X1)