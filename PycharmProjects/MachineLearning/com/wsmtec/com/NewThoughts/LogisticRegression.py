# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import time

start = time.time()
path = 'F:\workingData\\201711\\NotSeperatableData'
df = pd.read_csv(path +'/easy_seperate_data.csv')

data = np.array(df.drop('result',axis=1))
label = np.array(df['result'])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.3,random_state=1234)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000,penalty='l2',C=0.01)

lr.fit(xtrain,ytrain)
prob = lr.predict_proba(xtest)[:,1]
pred = lr.predict(xtest)

from sklearn import metrics
auc = metrics.roc_auc_score(ytest,prob)
fpr,tpr,_ = metrics.roc_curve(ytest,prob)
ks = tpr - fpr
con = metrics.confusion_matrix(ytest,pred)
recall = metrics.recall_score(ytest,pred)
f1 = metrics.f1_score(ytest,pred)
print('auc is ',auc)
print('ks = ',ks)
print('recall = ',recall)
print('f1 score =',f1)
print('confusion matrix ',con)
print('use time ',time.time() - start)