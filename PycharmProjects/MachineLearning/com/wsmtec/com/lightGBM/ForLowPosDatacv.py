# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics

t1 = time.time()
path = 'F:\workingData\\201709\data\Hive'
df_read = pd.read_csv(path+'/test.txt',sep='\t')
shuffle(df_read)
#add the colums
tmp = np.arange(1,94)
col = list()
for i in range(tmp.shape[0]):
    col.append(np.str(tmp[i]))
df_read.columns = col
df_read.drop(['1','92','93'],axis=1,inplace=True)
df = df_read



#use the for loop for cross validation
#split the pos data for 20% and 80%
auc_list = []
recall_list= []
ks_list = []
f1_list = []
confusion = []

for i in range(10):
    print('the epoch ',(i+1))
    # split the data for validation 20%
    # shuffle the data
    from sklearn.utils import shuffle
    shuffle(df)
    #use the train_test_split for the cvdf and df
    # cvdf = df_read[:400000]
    # df = df_read[400000:]
    cvdf,df = train_test_split(df,test_size=.8)

    # make the data for pos and neg
    pos = df.loc[df['91'] == 0]
    pos_data = np.array(pos.drop('91', axis=1))
    pos_label = pos['91']
    neg = df.loc[df['91'] == 1]
    neg_data = np.array(neg.drop('91', axis=1))
    neg_label = neg['91']
    # add the cv data and label
    cv_data = np.array(cvdf.drop('91', axis=1))
    cv_label = cvdf['91']

    pos_train,_,pos_train_label,_ = train_test_split(pos_data,pos_label,test_size=.6)
    data = np.concatenate((pos_train,neg_data),axis=0)
    label = np.concatenate((pos_train_label,neg_label),axis=0)
    # xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.2)
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

    # lgb_train = lgb.Dataset(xtrain, ytrain)
    # lgb_test = lgb.Dataset(xtest, ytest)
    lgb_train = lgb.Dataset(data, label)
    lgb_test = lgb.Dataset(cv_data, cv_label)

    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_test)
    prob = gbm.predict(cv_data)
    pred = np.array(pd.Series(prob).map(lambda x: 1 if x >= .5 else 0))
    auc = metrics.roc_auc_score(cv_label,prob)
    fpr,tpr,_ = metrics.roc_curve(cv_label,prob)
    ks = tpr - fpr
    recall = metrics.recall_score(cv_label,pred)
    f1 = metrics.f1_score(cv_label,pred)
    con = metrics.confusion_matrix(cv_label,pred)
    auc_list.append(auc)
    ks_list.append(ks.max())
    recall_list.append(recall)
    f1_list.append(f1)
    confusion.append(con.tolist())

print('auc list',auc_list)
print('ks list',ks_list)
print('recall list ',recall_list)
print('f1 list ',f1_list)
print('**********************')
print('auc avg ',np.mean(auc_list))
print('ks avg ',np.mean(ks_list))
print('recall avg ',np.mean(recall_list))
print('f1 avg',np.mean(f1_list))
print('confusion list ',np.array(confusion))
