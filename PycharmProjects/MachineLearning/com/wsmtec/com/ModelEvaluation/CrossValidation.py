# -*- coding:utf-8 -*-
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cv_score(estimator,data,label,cv=10):
    recall_score = make_scorer(metrics.recall_score,greater_is_better=True)
    precision_score = make_scorer(metrics.precision_score,greater_is_better=True)
    f1_score = make_scorer(metrics.f1_score,greater_is_better=True)
    auc_score = make_scorer(metrics.roc_auc_score,greater_is_better=True)

    recall_validation = cross_val_score(estimator, data, label, scoring=recall_score,cv=cv)
    precision_validation = cross_val_score(estimator, data, label, scoring=precision_score,cv=cv)
    f1_validation = cross_val_score(estimator, data, label, scoring=f1_score,cv=cv)
    auc_validation = cross_val_score(estimator, data, label, scoring=auc_score,cv=cv)

    print('use the 10-fold cross validation for the model, 10-fold recall = ', recall_validation.mean())
    print('use the 10-fold cross validation for the model, 10-fold precision = ', precision_validation.mean())
    print('use the 10-fold cross validation for the model, 10-fold f1_score = ', f1_validation.mean())
    print('use the 10-fold cross validation for the model, 10-fold auc = ', auc_validation.mean())