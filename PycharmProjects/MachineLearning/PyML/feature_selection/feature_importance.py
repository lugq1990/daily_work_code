# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:59:36 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier


def get_importance(X, y, feat_labels, estimator=None, n_estimators=1000):
    if not estimator:
        estr = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    else:
        estr = estimator
    estr.fit(X, y)
    importances = estr.feature_importances_
    indices = np.argsort(importances)[::1]
    
    plt.title('Feature Importances')
    plt.barh(range(X.shape[1]), sorted(importances), color='red', align='center')
    plt.yticks(range(X.shape[1]), feat_labels[indices])
    plt.show()


