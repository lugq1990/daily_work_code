# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#plot the best pca n_components using the grid search to choose the best n_components
#and plot the threshold line on the plot
#just for a more careful choosen n_components
#suggest first use the plot method to plot the all varibles variances
#then use the fine tuning the choosen space for the param(n_components)

def plot_pca(data):
    data = np.array(data)
    pca = PCA()
    pca.fit(data)
    variance_ratio = pca.explained_variance_ratio_
    plt.figure(1,figsize=(7,5))
    plt.clf()
    plt.plot(variance_ratio,linewidth=2)
    plt.axis('tight')
    plt.title('all variable variance')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()

def choose_best_components_pca_plot(estimator,n_components_list,param_list,data,label):
    estimator = estimator
    pca = PCA()
    pca.fit(data)
    variance_ratio = pca.explained_variance_ratio_
    pipe = Pipeline(steps=[('pca',pca),('estimator',estimator)])
    n_components = np.array(n_components_list)
    params = np.array(param_list)
    clf = GridSearchCV(pipe,dict(pca__n_components=n_components,estimator__C=params))
    clf.fit(data,label)
    plt.figure(1,figsize=(8,5))
    plt.clf()
    plt.plot(variance_ratio,linewidth=2)
    plt.axis('tight')
    plt.title('choosen the conponents')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    #plt.xticks(n_components_list)
    plt.axvline(clf.best_estimator_.named_steps['pca'].n_components,linestyle=':',label='n_components choosen')
    plt.legend(prop=dict(size=12))
    plt.show()


# from sklearn.datasets import load_digits
# x,y = load_digits(return_X_y=True)
# # plot_pca(x)
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# choose_best_components_pca_plot(lr,[10,12,14,16],[1,10],x,y)
