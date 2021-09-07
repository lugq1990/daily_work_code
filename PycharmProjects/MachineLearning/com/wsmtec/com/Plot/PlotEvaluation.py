# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV

def plot_learning_curve(estimator,data,label,param_name=['C','intercept_scaling'],params=[[1,10],[1,3,5,7]],cv=10):
    if(len(param_name)>1):
        param_grid = {}
        for i in range(len(param_name)):
            for j in range(len(params[i])):
                param_grid.setdefault(param_name[i],[]).append(params[i][j])
    else:
        param_grid = {param_name:params}
    print(param_grid)
    grid = GridSearchCV(estimator,param_grid=param_grid)
    grid.fit(data,label)
    clf = grid.best_estimator_
    skplt.estimators.plot_learning_curve(clf,data,label,cv=cv)
    plt.show()
    return clf

def plot_metrics(estimator,data,label):
    xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.2,random_state=1234)
    estimator.fit(xtrain,ytrain)
    pred = estimator.predict(xtest)
    prob = estimator.predict_proba(xtest)
    skplt.metrics.plot_confusion_matrix(ytest,pred)
    skplt.metrics.plot_roc_curve(ytest,prob)
    if(np.unique(label).shape[0]==2):
        skplt.metrics.plot_ks_statistic(ytest,prob)
    skplt.metrics.plot_precision_recall_curve(ytest,prob)
    plt.show()

# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils import shuffle
# x,y = load_iris(return_X_y=True)
# x,y = shuffle(x,y)
# lr = LogisticRegression()
# plot_learning_curve(lr,x,y)