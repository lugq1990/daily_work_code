# -*- coding:utf-8 -*-
"""This class is used for plot High Dimensional datasets, this including using PCA and TSNE for decomposite data
    to 2D or 3D, noted: TNSE and PCA maybe different for result curve.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style

class plotHighDimen(object):
    def __init__(self):
        pass

    # this is for using PCA to do decomposition, also can be used for dividing different classes.
    # this can also decide to plot which classes using 'plot_classes'
    def pcaPlot(self, data, label=None, plot_classes=None, figsize=(8, 6), title='Different Classes PCA Plot'):
        style.use('ggplot')

        # processing data to 2D
        pca = PCA(n_components=2)
        data_new = pca.fit(data).transform(data)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if label is None:
            ax.scatter(data_new[:, 0], data_new[:, 1])
        else:
            uni_label = np.unique(label)
            if plot_classes is not None:
                for p in plot_classes:
                    if p not in uni_label:
                        raise ValueError('Wanted plot classes %s is not in label!'%(str(p)))
            else:
                plot_classes = uni_label

            # loop for each class
            for cla in plot_classes:
                ax.scatter(data_new[label == cla, 0], data_new[label == cla, 1], label='class_'+str(cla))

        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.legend()
        plt.show()

    # Using TSNE to plot 2-D data scatter curve, noted: TSNE is based on iterations for data, so it maybe slower than PCA.
    def tsnePlot(self, data, label=None, plot_classes=None, n_iter=1000, figsize=(8, 6), title='Different Classes TSNE Plot'):
        style.use('ggplot')

        # processing data to 2D
        tsne = TSNE(n_components=2, random_state=1234, n_iter=n_iter)
        data_new = tsne.fit_transform(data)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if label is None:
            ax.scatter(data_new[:, 0], data_new[:, 1])
        else:
            uni_label = np.unique(label)
            if plot_classes is not None:
                for p in plot_classes:
                    if p not in uni_label:
                        raise ValueError('Wanted to plot class %s is not in label!'%(str(p)))
            else:
                plot_classes = uni_label

            # loop for each classes
            for classes in plot_classes:
                ax.scatter(data_new[label==classes, 0], data_new[label == classes, 1], label='class_'+str(classes))

        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.legend()

        plt.show()

    # this function is used for plot 3D scatter curve, including using TSNE or PCA to plot curve.
    # Default is using PCA to decomposite data to 3D.
    # Noted: by using TSNE, this will be slower than using PCA.
    def plot3D(self, data, label=None, plot_classes=None, use_tsne=False, tsne_iter=1000,
               figsize=(8, 6), title='3D classes Scatter Plot'):
        from mpl_toolkits.mplot3d import Axes3D
        style.use('ggplot')

        # use PCA to decomposite data to 3D
        if not use_tsne:
            pca = PCA(n_components=3)
            data_new = pca.fit_transform(data)
        else:
            tsne = TSNE(n_components=3, n_iter=tsne_iter)
            data_new = tsne.fit_transform(data)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if label is None:
            ax.scatter(data_new[:, 0], data_new[:, 1], data_new[:, 2])
        else:
            uni_label = np.unique(label)
            if plot_classes is not None:
                for p in plot_classes:
                    if p not in uni_label:
                        raise ValueError('Wanted to plot class %s is not in label!'%(str(p)))
            else:
                plot_classes = uni_label

            # loop for all wanted plot classes.
            for classes in plot_classes:
                ax.scatter(data_new[label==classes, 0], data_new[label==classes, 1], data_new[label==classes, 2],
                           label='class_'+str(classes))

        ax.set_title(title)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    iris = load_digits()
    x, y = iris.data, iris.target
    # p = plotHighDimen().tsnePlot(x, y, plot_classes=[0,1])
    plotHighDimen().plot3D(x, y, use_tsne=False)

