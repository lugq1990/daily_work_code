# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
x,y = iris.data,iris.target

lr = LogisticRegression()
lr.fit(x,y)

pred = lr.predict(x)

from sklearn.utils.multiclass import type_of_target
print(type_of_target(x))