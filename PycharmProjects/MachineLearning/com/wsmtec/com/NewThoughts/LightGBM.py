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

import lightgbm as lgb
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc', 'binary_error'],
    'num_leaves': 101,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0,
}
lgb_train = lgb.Dataset(xtrain,ytrain)
lgb_test = lgb.Dataset(xtest,ytest)

gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_test)
prob = gbm.predict(xtest,num_iteration=gbm.best_iteration)
pred = np.array(pd.Series(prob).map(lambda x:1 if x>=.5 else 0))

from sklearn import metrics
auc = metrics.roc_auc_score(ytest,prob)
fpr,tpr,_ = metrics.roc_curve(ytest,prob)
ks = tpr - fpr
recall = metrics.recall_score(ytest,pred)
confusion = metrics.confusion_matrix(ytest,pred)
print('auc=',auc)
print('ks=',ks.max())
print('recall=',recall)
print('confusion matrix ',confusion)
print('use time ',time.time() - start)
