# -*- coding:utf-8 -*-
""" This is just implement a class for abstracting sklearn algorithm object to train model,
    By using this class, I can also construct object with given params.
"""

class sklearnHelper(object):
    def __init__(self, clf, params=None, seed=0):
        params['random_state'] = seed
        self.clf = clf(**params)

    def fit(self, xtrain, ytrain):
        self.clf.fit(xtrain, ytrain)

    def score(self, xtest, ytest):
        sc = self.clf.score(xtest, ytest)
        print('Test datasets accuracy:', sc)
        return sc

    def predict(self, x):
        return self.clf.predict(x)

if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    iris = load_iris()
    x, y = iris.data, iris.target

    params = {'C':10}
    lr = sklearnHelper(clf=LogisticRegression, params=params)
    lr.fit(x, y)
    lr.score(x, y)
    print('LR prediction:', lr.predict(x)[:10])