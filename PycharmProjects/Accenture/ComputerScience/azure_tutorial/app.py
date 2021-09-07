# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
# from flask import Flask

# app = Flask(__name__)


# app.route('/hello')
# def hello():
#     return "Hi, this is lugq"

# if __name__ == '__main__':
#     hello()


import numpy as np
from joblib import delayed, Parallel
from itertools import chain
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
x, y =load_iris(return_X_y=True)

n_split = 4

def fit(C, xtrain, ytrain, xtest, ytest):
    lr.C = C
    lr.fit(xtrain, ytrain)

    score = lr.score(xtest, ytest)
    print("C: {}, score: {}".format(C, score))

# make random split
def split(n_split=n_split, test_size=.2):
    seed = np.random.RandomState(1)
    for _ in range(n_split):
        indices = seed.permutation(len(x))

        index = int((1 - test_size) * len(x))
        xtrain = x[:index]
        xtest = x[index:]
        ytrain = y[:index]
        ytest = y[index:]
        yield xtrain, ytrain, xtest, ytest


c_list = [1, 2, 3, 4]


Parallel(n_split)(delayed(fit)(c, xtrain, ytrain, xtest, ytest) for c, xtrain, ytrain, xtest, ytest in zip(*chain(c_list, split())))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LsiModel, Word2Vec
