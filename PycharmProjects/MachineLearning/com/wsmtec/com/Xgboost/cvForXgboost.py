# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
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

data = df.drop(['91'],axis=1)
label = df['91']

gbm = xgb.XGBClassifier(max_depth=8, silent=1, gamma=10, subsample=.8, colsample_bytree=.8, reg_alpha=1, reg_lambda=1,
                         objective='binary:logistic', learning_rate=.1)
#split the data for train and test
from sklearn.model_selection import train_test_split,cross_validate
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.3)
scoring = ['recall','precision','roc_auc','f1']
scores = cross_validate(gbm,data,label,scoring=scoring,cv=5,return_train_score=False)
recall = scores['test_recall'].mean()
precision = scores['test_precision'].mean()
auc = scores['test_roc_auc'].mean()
f1 = scores['test_f1'].mean()
print('recall avg=',recall)
print('precision avg=',precision)
print('auc=',auc)
print('f1=',f1)