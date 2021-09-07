# -*- coding:utf-8 -*-
"""this is to implement different preprocessing step with manually function"""
import numpy as np
from sklearn import preprocessing

x = np.random.random((3, 4))

# min-max scaler
def min_max_scaler(x):
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

# max abs scaler
def max_abs_scaler(x):
    return x / (np.abs(x).max(axis=0))

# standard
def standard(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def norm(x, norm='l2'):
    if norm == 'l1':
        return x / np.abs(x).sum(axis=1)[:, np.newaxis]
    elif norm == 'l2':
        return x / np.linalg.norm(x, axis=1)[:, np.newaxis]
    elif norm == 'max':
        return x / np.max(np.abs(x), axis=1)[:, np.newaxis]
    else:
        raise TypeError("couldn't process {} format".format(norm))



def closs_enough(x, y):
    return np.allclose(x, y)


if __name__ == '__main__':
    print('min-max:', np.allclose(preprocessing.MinMaxScaler().fit_transform(x), min_max_scaler(x)))
    print('max-abs:', closs_enough(preprocessing.MaxAbsScaler().fit_transform(x), max_abs_scaler(x)))
    print('standard:', closs_enough(preprocessing.StandardScaler().fit_transform(x), standard(x)))
    print('norm: ', closs_enough(preprocessing.Normalizer().fit_transform(x), norm(x)))
    print('l1 norm: ', closs_enough(preprocessing.Normalizer(norm='l1').fit_transform(x), norm(x, norm='l1')))
    print('max norm:', closs_enough(preprocessing.Normalizer(norm='max').fit_transform(x), norm(x, norm='max')))
