# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
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
            self.n_samples = int(float(self.N/100)*old_num)
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

t1 = time.time()
path = 'F:\workingData\\201709\data\Hive'
df = pd.read_csv(path+'/test.txt',sep='\t')
#add the shuffle for the data
from sklearn.utils import shuffle
df = shuffle(df)

tmp = np.arange(1,94)
col = list()
for i in range(tmp.shape[0]):
    col.append(np.str(tmp[i]))
df.columns = col
df.drop(['1','92','93'],axis=1,inplace=True)


#make the data for pos and neg
pos = df.loc[df['91']==0]

#random get the pos data for 80000 items
pos = pos.sample(n=610000,replace=False)
pos_data = np.array(pos.drop('91',axis=1))
pos_label = pos['91']
neg = df.loc[df['91']==1]
neg_data = np.array(neg.drop('91',axis=1))
neg_label = neg['91']
print('start generate data')
start = time.time()

#use my_over_sampling
#my_over = my_over_sampling(samples=neg_data,num=150000).over_sampling()
#my_over = SMOTE(neg_data,N=25).over_sample()
#my_over_label = np.ones((my_over.shape[0]))
print('finished!')
end = time.time()
print('make the data use ',end-start,'s')
#the produced and original data combined
#data_neg = np.concatenate((neg_data,my_over),axis=0)
#label_neg = np.concatenate((neg_label,my_over_label),axis=0)

#all combined the produced data and the original data
#make the down-sampling for the pos data

data = np.concatenate((neg_data,pos_data),axis=0)
label = np.concatenate((neg_label,pos_label),axis=0)

#the original all data
df_data = df.drop(['91'],axis=1)
df_label = df['91']
df_data = np.array(df_data)
df_label = np.array(df_label)

#make the data for train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data,label)

params = {
    'boosting_type': 'gbdt',
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

gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_test)
prob = gbm.predict(df_data)

#evaluate the model
from sklearn import metrics
auc = metrics.roc_auc_score(df_label,prob)
fpr,tpr,_ = metrics.roc_curve(df_label,prob)
ks = tpr - fpr
pred = np.array(pd.Series(prob).map(lambda x:1 if x>=.5 else 0))
f1 = metrics.f1_score(df_label,pred)
con = metrics.confusion_matrix(df_label,pred)
recall = metrics.recall_score(df_label,pred)
print("auc=",auc)
print("ks=",ks.max(axis=0))
print('f1 score',f1)
print('recall',recall)
print('confusion matrix',con)
t2 = time.time()
print("the all line use ",t2-t1,"s")

