# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import random
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

style.use('fivethirtyeight')
iris = load_digits()
x, y =iris.data, iris.target
x, y = shuffle(x, y)
#x = PolynomialFeatures(degree=3).fit_transform(x)
x = PCA(n_components=40).fit_transform(x)

def plot(x, y, ax=None, title=None):
    if x.shape[1] > 2:
        x_pca = PCA(n_components=2).fit_transform(x)
    else: x_pca = x
    for i in range(len(np.unique(y))):
        if ax is None or title is None:
            plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], label='class_'+np.str(i+1))
            plt.title('Clustering Result')
        else:
            ax.scatter(x_pca[y == i, 0], x_pca[y == i, 1], label='class_'+np.str(i+1))
            ax.set_title(title)
    plt.legend()

# plot_iris(x, y)
def kmeans(x, k=3, iter=10, tol=.001):
    # first get the random index as the center point,
    # because I do not want get the same index, use the random.sample()
    rand_index = random.sample(range(len(x)), k)
    init_center = x[rand_index, :]
    center_list = list()
    for ii in range(iter):
        res_list = list()
        medium_center = list()
        for i in range(len(x)):
            medium_dis = list()
            # because I do not want get the same dataset, so I just get satisfied datasets
            if ii == 0:
                x_t = np.ones(x.shape[0], dtype=bool)
                x_t[rand_index] = False
                x_new = x[x_t,:]
            else: x_new = x

            # because of first time, we have get K data from original datasets, so we can not get K data
            if ii == 0 and i >= len(x_new) - 1 :
                continue

            for j in range(k):
                d = np.linalg.norm(init_center[j, :] - x_new[i, :])
                medium_dis.append(d)

            res_list.append([x_new[i, :], np.argmax(medium_dis)])
        # this is to compute the new center point
        for m in range(k):
            # first to know what different classes data
            sati = np.array(res_list)[:, 1] == m
            if (sum(sati) == 0) and ii > 0:
                warnings.warn('During fitting, there is one class has no dataset!')
                medium_center.append(center_list[ii - 1][m])
                continue

            # compute the each class center
            new_center = np.mean(np.array(res_list)[:, 0][sati], axis=0)
            medium_center.append(new_center.tolist())
        # just append the center result to center list
        center_list.append(medium_center)
        # Important! change the original dataset center
        init_center = np.array(medium_center)
        # compare whether or not the center does not change
        if ii > 1 and (np.linalg.norm(np.array(center_list[ii]) - np.array(center_list[ii - 1])) <= tol):
            print('the center does not change any more! \nFinished!')
            break

    return np.array(res_list)

# res = kmeans(x, k=5, iter=50)
# plot(x, res[:, 1])


# This is used to use kmeans to do clustering
km = KMeans(n_clusters=10).fit(x)
fig, ax = plt.subplots(1, 2, figsize=(12,8))
plot(km.transform(x), y, ax[0], 'True result')
plot(km.transform(x), km.labels_, ax[1], 'prediction result')
plt.show()
