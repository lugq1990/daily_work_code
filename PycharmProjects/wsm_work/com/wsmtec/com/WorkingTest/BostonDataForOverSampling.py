# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class my_over_sampling:
    def __init__(self,samples,num=10):
        self.samples = samples
        self.num = num
        self.n_samples,self.n_cols = samples.shape

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
        dis = np.zeros((1,self.n_cols))
        for i in range(num_sam):
            dif = self.centers - self.samples[i]
            dis += dif
        avg_dis = dis/num_sam
        #print(avg_dis[0])
        return avg_dis

boston = load_boston()
x,y = boston.data,boston.target
pos = x[y>40]
neg = x[y<=40]
y_pos = np.ones((pos.shape[0]))
y_neg = np.zeros((neg.shape[0]))

#combine the data to be one
data = np.concatenate((pos,neg),axis=0)
label = np.concatenate((y_pos,y_neg),axis=0)

#split the data to train and test
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.4)

#use the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)
pred = dtc.predict(xtest)
prob = dtc.predict_proba(xtest)

#compute the recall ,auc and confusion matrix
from sklearn import metrics
auc = metrics.roc_auc_score(ytest,prob[:,1])
recall = metrics.recall_score(ytest,pred)
confusion_matrix = metrics.confusion_matrix(ytest,pred)
fpr,tpr,_ = metrics.roc_curve(ytest,prob[:,1])
print("auc=",auc)
print("recall=",recall)
print("confusion matrix",confusion_matrix)



#use the over-sampling for the data
my_over_pos = my_over_sampling(pos,num=444).over_sampling()
my_over_label = np.ones((my_over_pos.shape[0]))
data_over = np.concatenate((data,my_over_pos),axis=0)
label_over = np.concatenate((label,my_over_label),axis=0)
#split the data
xtrain_o,xtest_o,ytrain_o,ytest_o = train_test_split(data_over,label_over,test_size=.4)
dtc.fit(xtrain_o,ytrain_o)
pred_o = dtc.predict(xtest_o)
prob_o = dtc.predict_proba(xtest_o)
auc_o = metrics.roc_auc_score(ytest_o,prob_o[:,1])
recall_o = metrics.recall_score(ytest_o,pred_o)
con_o = metrics.confusion_matrix(ytest_o,pred_o)
print("auc_o=",auc_o)
print("recall_o=",recall_o)
print("confusion_o ",con_o)

# #use the imblanced-learn SMOTE
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(kind='borderline1')
# data_smote,label_smote = smote.fit_sample(data,label)
# xtrain_smote,xtest_smote,ytrain_smote,ytest_smote = train_test_split(data_smote,label_smote,test_size=.4)
# dtc.fit(xtrain_smote,ytrain_smote)
# pred_smote = dtc.predict(xtest_smote)
# prob_smote = dtc.predict_proba(xtest_smote)
# auc_smote = metrics.roc_auc_score(ytest_smote,prob_smote[:,1])
# recall_smote = metrics.recall_score(ytest_smote,pred_smote)
# con_smote = metrics.confusion_matrix(ytest_smote,pred_smote)
# print("auc_smote=",auc_smote)
# print("recall_smote=",recall_smote)
# print("confusion matrix",con_smote)