# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from collections import Counter

iris = load_iris()
x, y = iris.data, iris.target
lr = LogisticRegression()
lr.fit(x, y)
# this is model prediction prob result.
prob = lr.predict_proba(x)
tmp = np.random.random((len(y), 3))

# this will work.
skplt.metrics.plot_roc(y, prob)

print('Different Classes Count res: ', Counter(y))
# Because label is 3-classes, but given object result 'tmp' is 4D, so
# this failed, raise error: ValueError: Found input variables with inconsistent numbers of samples
skplt.metrics.plot_roc(y, tmp)
plt.show()