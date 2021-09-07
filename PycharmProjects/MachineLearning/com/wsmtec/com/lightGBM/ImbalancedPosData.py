# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import time

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

t1 = time.time()
path = 'F:\workingData\\201709\data\Hive'
df = pd.read_csv(path+'/test.txt',sep='\t')
#add the colums
tmp = np.arange(1,94)
col = list()
for i in range(tmp.shape[0]):
    col.append(np.str(tmp[i]))
df.columns = col
df.drop(['1','92','93'],axis=1,inplace=True)

#shuffle the data
from sklearn.utils import shuffle
shuffle(df)

#make the data for pos and neg
pos = df.loc[df['91']==0]
pos_data = np.array(pos.drop('91',axis=1))
pos_label = pos['91']
neg = df.loc[df['91']==1]
neg_data = np.array(neg.drop('91',axis=1))
neg_label = neg['91']

#use the my_over
# my_over = my_over_sampling(np.array(neg_data),num=50000).over_sampling()
# my_over_label = np.ones((my_over.shape[0],))
# neg_data = np.concatenate((neg_data,my_over),axis=0)
# neg_label = np.concatenate((neg_label,my_over_label),axis=0)

#make the pos data for two parts
from sklearn.model_selection import train_test_split
pos_1_data,pos_2_data,pos_1_label,pos_2_label = train_test_split(pos_data,pos_label,test_size=.8,random_state=1234)
#combine the pos and neg data
data = np.concatenate((neg_data,pos_1_data),axis=0)
label = np.concatenate((neg_label,pos_1_label),axis=0)

#split the data for train and test
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.2)

#construct the model
params = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': ['auc','binary_error'],
    'num_leaves': 101,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0,
}

lgb_train = lgb.Dataset(xtrain,ytrain)
lgb_test = lgb.Dataset(xtest,ytest)

gbm = lgb.train(params,lgb_train,num_boost_round=10,valid_sets=lgb_test)
prob = gbm.predict(xtest)
pred = np.array(pd.Series(prob).map(lambda x:1 if x>=.5 else 0))

#evaluate the model
from sklearn import metrics
# auc = metrics.roc_auc_score(ytest,prob)
# fpr,tpr,_ = metrics.roc_curve(ytest,prob)
# ks = tpr - fpr
# f1 = metrics.f1_score(ytest,pred)
# con = metrics.confusion_matrix(ytest,pred)
# recall = metrics.recall_score(ytest,pred)
# print("auc=",auc)
# print("ks=",ks.max(axis=0))
# print('f1 score',f1)
# print('recall',recall)
# print('confusion matrix',con)
# t2 = time.time()
# print("the all line use ",t2-t1,"s")

#now with the 80% pos data and 20% neg data
data_80 = np.concatenate((pos_data,neg_data),axis=0)
label_80 = np.concatenate((pos_label,neg_label),axis=0)
xtrain_80,xtest_80,ytrain_80,ytest_80 = train_test_split(data_80,label_80,test_size=.2)
lgb_train_80 = lgb.Dataset(xtrain_80,label=ytrain_80)
lgb_test_80 = lgb.Dataset(xtest_80,label=ytest_80)
gbm_80 = lgb.train(params,lgb_train_80,num_boost_round=10,valid_sets=lgb_test_80)
prob_80 = gbm_80.predict(xtest_80)
pred_80 = np.array(pd.Series(prob_80).map(lambda x:1 if x>=.5 else 0))

#evaluate the model
# auc2 = metrics.roc_auc_score(ytest_80,prob)
# fpr2,tpr2,_ = metrics.roc_curve(ytest_80,prob)
# ks2 = tpr2 - fpr2
# f1_2 = metrics.f1_score(ytest_80,pred_80)
# con = metrics.confusion_matrix(ytest_80,pred_80)
# recall = metrics.recall_score(ytest_80,pred_80)



#combine the two model
all_data = df.drop(['91'],axis=1)
all_label = df['91']
prob1,prob2 = gbm.predict(all_data),gbm_80.predict(all_data)
pred1 = np.array(pd.Series(prob1).map(lambda x:1 if x>=.5 else 0))
pred2 = np.array(pd.Series(prob2).map(lambda x:1 if x>=.5 else 0))

auc2 = metrics.roc_auc_score(all_label,prob1)
fpr2,tpr2,_ = metrics.roc_curve(all_label,prob1)
ks2 = tpr2 - fpr2
f1_2 = metrics.f1_score(all_label,pred1)
con2 = metrics.confusion_matrix(all_label,pred1)
recall2 = metrics.recall_score(all_label,pred1)

auc1 = metrics.roc_auc_score(all_label,prob2)
fpr1,tpr1,_ = metrics.roc_curve(all_label,prob2)
ks1 = tpr2 - fpr2
f1_1 = metrics.f1_score(all_label,pred2)
con1 = metrics.confusion_matrix(all_label,pred2)
recall1 = metrics.recall_score(all_label,pred2)

auc = (auc1+auc2)/2
ks = (ks1+ks2)/2
f1 = (f1_1+f1_2)/2
recall = (recall1+recall2)/2


print("auc=",auc)
print("ks=",ks.max(axis=0))
print('f1 score',f1)
print('recall',recall)
print('confusion matrix1',con1)
print('confusion matrix2',con2)