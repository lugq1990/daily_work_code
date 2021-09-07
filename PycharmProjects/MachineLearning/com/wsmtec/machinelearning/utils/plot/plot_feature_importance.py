# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature(x,importance,imp_lable):
    plt.title('importance title')
    plt.barh(range(x.shape[1]),sorted(importance),color='red',align='center')
    imp = np.argsort(importance)[::1]
    # plt.yticks(range(x.shape[1]),imp_lable[imp])
    plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris = load_iris()
#x,y= iris.data, iris.target
x = iris.data
y = iris.target
label = ['a','b','c','d']
dtc = DecisionTreeClassifier()
dtc.fit(x,y)
imp = dtc.feature_importances_
plot_feature(x,imp,label)