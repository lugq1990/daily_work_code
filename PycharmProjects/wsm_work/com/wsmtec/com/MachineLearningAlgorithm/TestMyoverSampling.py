# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import logging
logging.getLogger().setLevel(logging.DEBUG)

from sklearn.neighbors import NearestNeighbors
class SMOTE:
    def __init__(self,samples,N=10,k=5):
        self.samples = samples
        self.N = N
        self.k = k
        self.n_samples,self.n_attrs = samples.shape
        #self.num = num

    def over_sample(self):
        if self.N < 100:
            old_num = self.n_samples
            self.n_samples = int(float(self.N)/100*old_num)
            keep = np.random.permutation(old_num)[:self.n_samples]
            new_samples = self.samples[keep]
            self.samples = new_samples
            self.N = 100
        N = int(self.N/100)
        #print(self.samples.shape)
        self.new_index = 0
        self.result = np.zeros((self.n_samples*N,self.n_attrs))
        #print(self.result.shape)

        nb = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray = nb.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print("nnarray",nnarray)
            self.__compute(N,nnarray)
        return self.result
    def __compute(self,N,nnarray):
        for i in range(N):
            nn = np.random.randint(1,self.k)
            diff = self.samples[nnarray[nn]] - self.samples[i]
            gap = np.random.rand(1,self.n_attrs)
            self.result[self.new_index] = self.samples[i] + gap.flatten()*diff
            #print(self.result[i])
            self.new_index +=1

class my_over_sampling:
    def __init__(self,samples,num=10,pos_samples=None):
        self.samples = samples
        self.num = num
        self.n_samples,self.n_cols = samples.shape
        self.pos_samples = pos_samples

    def over_sampling(self):
        self.index = 0
        self.result = np.zeros((self.num,self.n_cols))
        self.centers = np.mean(self.samples,axis=0)
        #print("center ",self.centers)
        dis = self.__computeDis()[0]
        for i in range(self.num):
            rotio = np.random.random()
            obj_choose = np.random.randint(self.n_samples)
            #add the guassian mixer
            mu, sigma = 0,0.1
            noise_rotio = np.random.normal(mu,sigma,size=(1,self.n_cols))[0]
            #print("!!",noise_rotio[0])
            self.result[i] = self.samples[obj_choose] + rotio*dis + noise_rotio
        return self.result

    #compute the center for the data
    def __computeDis(self):
        num_sam = self.samples.shape[0]
        #add the pos data center and distance
        if(self.pos_samples is not None):
            print("now in the pos data !")
            num_sam = self.pos_samples.shape[0]
            self.n_cols = self.pos_samples.shape[1]
            self.pos_center = np.mean(self.pos_samples,axis=0)
            dis_pos = np.zeros((1,self.n_cols))
            for j in range(num_sam):
                dif_pos = self.pos_center - self.pos_samples[j]
                dis_pos += dif_pos
            avg_pos_dis = dis_pos/num_sam
            return avg_pos_dis
        else:
        #use the original data center and distance
            dis = np.zeros((1,self.n_cols))
            for i in range(num_sam):
                dif = self.centers - self.samples[i]
                dis += dif
            avg_dis = dis/num_sam
            #print(avg_dis[0])
            return avg_dis

path = 'F:\workingData\\201709\data'
df = pd.read_csv(path+'/payday_1.csv')
df.drop('id_card',axis=1,inplace=True)
df.fillna(0,axis=0,inplace=True)
data = df.drop('Result',axis=1)
label = df['Result']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain_split,ytest_split = train_test_split(data,label,test_size=.2)
#change the ytrain series to dataframe
ytrain = pd.DataFrame(ytrain_split)
ytest = np.array(ytest_split).reshape(-1,1)
#make the neg data and label
neg_train_label = ytrain.loc[ytrain['Result']==1]
neg_train_index = ytrain.loc[ytrain['Result']==1].index
#add the pos data and label
pos_train_label = ytrain.loc[ytrain['Result']==0]
pos_train_data = xtrain.loc[pos_train_label.index]
#now get the neg train data
neg_train = xtrain.loc[neg_train_index]
#now make the neg data

#compute the time
import time
start = time.time()
neg_train_array = np.array(neg_train)
pos_train_array = np.array(pos_train_data)
my_over = my_over_sampling(neg_train_array,num=150000,pos_samples=pos_train_array).over_sampling()

#use the smote algorithm to make the data
#my_over = SMOTE(neg_train_array,N=200).over_sample()

end = time.time()
print("the over sampling data time is ",end - start)
my_over_label = np.ones((my_over.shape[0],1))
#concate the data to be one, that is the all neg data
new_neg_data = np.concatenate((neg_train,my_over),axis=0)
new_neg_label = np.concatenate((neg_train_label,my_over_label),axis=0)

#combine the neg and pos data to be one
data_train = np.concatenate((pos_train_data,new_neg_data),axis=0)
label = np.concatenate((pos_train_label,new_neg_label),axis=0)

#make the label to be just n-rows
label = label.reshape(-1)
ytest = ytest.reshape(-1)
#use the lightGBM to train the data

import lightgbm as lgb
#the original lgb data
lgb_train = lgb.Dataset(data_train,label=label)
#lgb_train = lgb.Dataset(xtrain,label=ytrain_split)
lgb_test = lgb.Dataset(xtest,label=ytest_split)
params = {'task':'train',
          'boosting_type':'gbdt',
          'objective':'binary',
          'metric':{'auc','binary_error'},
          'num_leaves':101,
          'learning_rate':.01,
          'feature_fraction':.9,
          'bagging_fraction':.8,
          'bagging_freq':5,
          'verbose':0
        }
print("start train the model!")
train_start = time.time()
#the original lgb
gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_test,early_stopping_rounds=10)

#add the cross validation for the lightGBM
# param = {'num_leaves':[70,80,90,100]}
# from sklearn.model_selection import GridSearchCV
# CV = GridSearchCV(estimator=lgb.LGBMClassifier(),param_grid=param,cv=3)
# gbm = gbm.fit(data_train,label)
pred = gbm.predict(xtest)
#evaluate the model
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest,pred)
print("the added new auc is ",auc)
from sklearn import metrics
fpr,tpr,_= metrics.roc_curve(ytest,pred)
ks = tpr - fpr
print("the ks is ",ks.max(axis=0))
#compute the f1-score
tmp = pd.Series(pred).map(lambda x:1 if x>=.5 else 0)
pred_not_prob = np.array(tmp)
f1_score = metrics.f1_score(ytest,pred_not_prob)
print("the f1_score is ",f1_score)
recall = metrics.recall_score(ytest,pred_not_prob)
print("the recall score is ",recall)
con_matrix = metrics.confusion_matrix(ytest,pred_not_prob)
print("the confusion matrix is ",con_matrix)

# now use the logistic regression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=20)
# lr.fit(data,label)
# pred = lr.predict_proba(xtest)[:,1]
# from sklearn.metrics import roc_auc_score
# print("the auc is ",roc_auc_score(ytest,pred))


#use the random forest adding cross validation
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn import metrics
#
# rfc = RandomForestClassifier()
# params = {'max_depth':np.arange(5,10)}
# cv = GridSearchCV(estimator=rfc,param_grid=params,cv=5)
# model = cv.fit(data_train,label)
# pred = model.predict(xtest)
# prob = model.predict_proba(xtest)
# recall = metrics.recall_score(ytest_split,pred)
# auc = metrics.roc_auc_score(ytest_split,prob[:,1])
# fpr,tpr,_ = metrics.roc_curve(ytest_split,prob[:,1])
# ks = tpr - fpr
# f1 = metrics.f1_score(ytest_split,pred)
# con = metrics.confusion_matrix(ytest_split,pred)
# print('the auc is ',auc)
# print('ks =',ks.max())
# print('f1_score=',f1)
# print('recall_score=',recall)
# print('consusion matrix',con)
