# -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = 'F:\workingData\\201709\data'
#df1 = pd.read_csv(path+'/payday_1.csv',encoding='gb2312')
df = pd.read_csv(path+'/test1706_2.csv',encoding='gb2312')
# x_train = df1.drop(['id_card', 'Result', 'id_5m_qy_n', 'id_4m_qy_n',
#                     'id_2m_order_avg', 'id_4m_order_avg', 'id_6m_order_avg', 'id_3m_order_avg', 'id_5m_order_avg',
#                     'id_1m_order_avg', 'id_tel_n', 'id_5m_loanM_avg', 'relat_id_same', 'id_1m_uid_avg',
#                     'id_4m_loanM_avg',
#                     'id_6m_loanM_avg', 'id_5m_order_n', 'id_2m_uid_avg', 'id_3m_loanM_avg'], axis=1)
#y_train = df1.Result
data = df.drop(['id_card', 'result', 'id_5m_qy_n', 'id_4m_qy_n','installment',
                   'id_2m_order_avg', 'id_4m_order_avg', 'id_6m_order_avg', 'id_3m_order_avg', 'id_5m_order_avg',
                   'id_1m_order_avg', 'id_tel_n', 'id_5m_loanM_avg', 'relat_id_same', 'id_1m_uid_avg',
                   'id_4m_loanM_avg',
                   'id_6m_loanM_avg', 'id_5m_order_n', 'id_2m_uid_avg', 'id_3m_loanM_avg'], axis=1)
label = df.result
data= np.array(data)
label = np.array(label)

xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.2)

#train and test data
train_data = xgb.DMatrix(xtrain,label=ytrain)
test_data = xgb.DMatrix(xtest,label=ytest)
watch_list = [(test_data, 'eval'), (train_data, 'train')]
param = {
'max_depth': 8,
'eta': 0.05,
'silent': 1,
'gamma':0,
'subsample':0.8,
'colsample_bytree' : 0.8,
'alpha':1,
'lambda':1,
'objective': 'binary:logistic',
'min_child_weight': 5,
'save_period':10,
'eval_metric':['auc','error'],
'evals_result':{},
'learning_rates':0.1
}
print("start train the model")
model = xgb.train(param,train_data,num_boost_round=100,evals=watch_list)

pred = model.predict(xtest,ntree_limit=model.best_ntree_limit)
auc = metrics.roc_auc_score(ytest,pred)
print("auc is ",auc)
