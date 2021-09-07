# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

style.use('fivethirtyeight')

iris = load_iris()
x, y = iris.data, iris.target
x, y = shuffle(x, y)
xtrain, ytrain = x[:-1, :], y[:-1]
xtest, ytest = x[-1, :], y[-1]

# first just to plot the classes
x_pca = PCA(n_components=2).fit(x).transform(x)[:-1, :]
for i in range(3):
    plt.scatter(x_pca[ytrain == i, 0], x_pca[ytrain == i, 1], label='class_'+np.str(i))
plt.scatter(x_pca[-1, 0], x_pca[-1, 1], label='new_class')
plt.legend()
plt.show()

# this is the KNN algorithm, also with the confidence = #most appears classes num/K
def kneighbor(data, label, new_feature, k=3):
    if (len(data) < k):
        warnings.warn('The dataset is not big enough!')
        return 1
    res_list = list()
    for i in range(len(data)):
        dis = np.linalg.norm(np.array(data[i, :]) - np.array(new_feature))
        res_list.append([dis, label[i]])
    voted = [m[1] for m in sorted(res_list)[:k]]
    counted = Counter(voted).most_common(1)[0][0]
    confidence = Counter(voted).most_common(1)[0][1] / k
    return counted, confidence

counted, confidence = kneighbor(xtrain, ytrain, xtest, k=3)
print('The new features belongs to classes: %d with confidence:%.4f, and true classes: %d'%(counted, confidence, ytest))

# bellow is just used to compute the KNN algorithm accuracy for test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=1234)
correct = 0
total = 0
for i in range(len(xtest)):
    re, _ = kneighbor(xtrain, ytrain, xtest[i, :], k=3)
    if re == ytest[i]:
        correct +=1
    total += 1
print('The KNN I write accuracy = {0:.4f}'.format(float(correct/total)))

