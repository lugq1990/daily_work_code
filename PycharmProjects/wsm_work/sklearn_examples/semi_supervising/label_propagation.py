# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.semi_supervised import label_propagation
from sklearn.utils import shuffle

x,y = load_iris(return_X_y=True)
x = x[:,:2]
x,y = shuffle(x,y)

rng = np.random.RandomState(0)

h = .02

y_30 = np.copy(y)
y_30[rng.rand(len(y))<.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y))<.5] = -1

ls30 = (label_propagation.LabelPropagation().fit(x,y_30),y_30)
ls50 = (label_propagation.LabelPropagation().fit(x,y_50),y_50)
clf_svm = (SVC(kernel='rbf').fit(x,y),y)

x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

titles = ['30%','50%','svm']

color = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

for i,(clf,ytrain) in enumerate((ls30,ls50,clf_svm)):
    plt.subplot(1,3,i+1)
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.Paired)
    plt.axis('off')

    colors = [color[y] for y in ytrain]
    plt.scatter(x[:,0],x[:,1],c=colors,edgecolors='black')

    plt.title(titles[i])

plt.show()