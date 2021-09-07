# -*- coding:utf-8 -*-
"""
    This is just used to plot the different clustering algorithm for iris and digits problems,
    here I have used the KMeans and GaussianMixture model to build it.
    Bacuase of I want to plot the result, so I use the PCA to decomposite the datasets to 2-D
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.datasets import load_digits

iris = load_iris()
x, y = iris.data, iris.target
# digits = load_digits()
# x, y = digits.data, digits.target

n_classes = np.unique(y).shape[0]

pred_kmeans = KMeans(n_clusters=n_classes).fit(x).predict(x)
pred_gmm = GaussianMixture(n_components=n_classes).fit(x).predict(x)
pred_lda = LatentDirichletAllocation(n_components=n_classes).fit(x).transform(x)
pred_lda = np.argmax(pred_lda, axis=1)
pred_affinity = AffinityPropagation().fit(x).predict(x)
pred_meanshift = MeanShift().fit(x).predict(x)
pred_dbscan = DBSCAN().fit_predict(x)

pred_list = [pred_gmm, pred_kmeans, pred_affinity,pred_meanshift, pred_dbscan]
model_list = ['kmeans', 'gaussian mixture', 'affinityPropogation', 'meanshift','DBSCAN']

# because I want to plot the data, so I have to use the PCA to decomposite the data to 2-D
x_pca = PCA(n_components=2).fit(x).transform(x)

def clusterRePlot(pred_list = pred_list, model_list = model_list):
    n_plots = len(pred_list) + 1
    # this is the generated result label
    label_list = list()
    for m in range(n_plots):
        for n in range(np.unique(y).shape[0]):
            if m == 0:
                label_list.append('true_' + np.str(n))
            else:
                label_list.append('pred_'+ np.str(n))
    label_list = np.asarray(label_list).reshape(n_plots, np.unique(y).shape[0])
    fig, ax = plt.subplots(1, n_plots, figsize=(14, 8))
    # start to plot the result
    for i in range(n_plots):
        for j in range(np.unique(y).shape[0]):
            if i == 0:
                ax[i].scatter(x_pca[y == j][:, 0], x_pca[y == j][:, 1], label=label_list[i, j])
                ax[i].set_title('True distribution')
            else:
                ax[i].scatter(x_pca[pred_list[i-1] == j][:, 0], x_pca[pred_list[i-1] == j][:, 1], label=label_list[i, j])
                ax[i].set_title(model_list[i-1])
    plt.legend()
    plt.show()
clusterRePlot()