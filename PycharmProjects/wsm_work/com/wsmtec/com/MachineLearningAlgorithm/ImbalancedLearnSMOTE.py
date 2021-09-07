# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import time

start = time.time()
path = 'F:\workingData\\201709\data'
df = pd.read_csv(path+'/payday_1.csv')
df.drop('id_card',axis=1,inplace=True)
df.fillna(0,axis=0,inplace=True)
data = df.drop('Result',axis=1)
label = df['Result']
from sklearn.model_selection import train_test_split
data = np.array(data)[:,:2]
label = np.array(label)
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.2)

from imblearn.over_sampling import SMOTE
smote = SMOTE(kind='svm')
xtrain_new,ytrain_new = smote.fit_sample(xtrain,ytrain)

print(xtrain_new.shape)
print(xtrain.shape)
end = time.time()
print("the data generate time is ",end-start)
