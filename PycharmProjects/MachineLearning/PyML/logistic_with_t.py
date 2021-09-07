# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:57:51 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from PyML.model import Logistic
from PyML.chart import draw_learning_curve, draw_validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, f1_score
from PyML.chart import roc_ks_curve


#设置路径
BASE_DIR = os.path.dirname(__file__) 
FILE_DIR = os.path.join(BASE_DIR,'data01.csv') 
RESULT_DIR = os.path.join(BASE_DIR,'logit_resutl.csv') 

dat = pd.read_csv(FILE_DIR)
features = dat.columns[2:]
y = np.array(dat.overdue)
y0 = np.zeros(y.shape[0])
y1 = np.zeros(y.shape[0])
y0[y==0] = 1
y1[y>0] = 1
X = np.mat(dat.iloc[:,2:21])

features = dat.columns[2: (X.shape[1]+2)]
logit = Logistic.Logit()
logit.fit(X, y, features)
logit.print_result()

'''
#logit.result.to_csv(RESULT_DIR, encoding="gb18030", index=False)
#drawn_data = logit.get_feat_names(1.64)
#drawn_data.remove("constant")
#X2 = np.mat(dat[drawn_data])
#X2 = np.mat(StandardScaler().fit_transform(X2))

#logit = LogisticRegression()

logit = LogisticRegression(solver="liblinear", max_iter=5000, 
                           #intercept_scaling=100, 
                           penalty="l2", C=0.1,
                           class_weight={0: 0.2, 1: 0.8}
                           )


#logit = LogisticRegression(penalty="l2", random_state=0)
param_name = "C"
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
draw_validation_curve.DRAW(X, y, param_name="C", param_range=param_range, clf=logit, ylim=[0.55, 0.65])
#draw_learning_curve.DRAW(X, y, clf=logit, ylim=[0.55, 0.68])


accu = 0
recall = 0
roc = 0
f1 = 0
X_, X_validate, y_, y_validate = train_test_split(X, y,  test_size=.1)
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_, y_,  test_size=.2)
    logit.fit(X_train, y_train)

    pred_y = logit.predict(X_test)
    y_predprob = logit.predict_proba(X_test)[:,1]
    acci = accuracy_score(y_test, pred_y)
    recalli = recall_score(y_test, pred_y)
    roci = roc_auc_score(y_test, y_predprob)
    f1i = f1_score(y_test, pred_y)
    accu += acci
    recall += recalli
    roc += roci
    f1 += f1i
    print np.sum(y_test), np.sum(pred_y), acci, recalli, f1i, roci

print
print "==============================="
print "Training result:"
print "   avg_accur: %f" %(accu/10.)
print "   avg_recall: %f" %(recall/10.)
print "   avg_roc: %f" %(roc/10.)
print "   avg_F1: %f" %(f1/10.)
print "==============================="
print


y_prob = logit.predict_proba(X_validate)
roc_ks_curve.DRAW(y_validate, y_prob)
'''