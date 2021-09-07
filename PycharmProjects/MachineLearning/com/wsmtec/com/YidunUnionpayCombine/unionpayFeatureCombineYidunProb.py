# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt

path = 'F:\workingData\\201801\yidun_union'
df = pd.read_excel(path+'/combine_union_features.xlsx')
#drop the not matched data
df.dropna(axis=0,inplace=True)
cate_col = 'CDTB150'
df.drop(cate_col,axis=1,inplace=True)


df_np = np.array(df)
id = df_np[:,0]
data = df_np[:,1:df_np.shape[1]-3]
data_prob = df_np[:,df_np.shape[1]-2].reshape(-1,1)
label = df_np[:,df_np.shape[1]-1].astype(np.int)

#impute for the -9999999 data to use the mean data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=-9999999,axis=1,strategy='mean')
data_imp = imp.transform(data)

#first standard the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data_imp)
data_scaler = scaler.transform(data_imp)

#decomposition for the unionpay data
#according to the PCA plot, get the 50 components
from sklearn.decomposition import PCA
pca = PCA(n_components=25).fit(data_scaler)
data_pca = pca.transform(data_scaler)

data_combine = np.concatenate((data_pca,data_prob),axis=1)

#split the data to train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data_combine,label,test_size=.2,random_state=1234)

#first use the Logistic Regression to train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=10,penalty='l2',random_state=1234,class_weight={0:.2,1:0.7})

#plot the learning curve for the lr
# skplt.estimators.plot_learning_curve(lr,data_pca,label,cv=10)
# plt.show()

# lr.fit(xtrain,ytrain)
# pred = lr.predict(xtest)
# prob = lr.predict_proba(xtest)

#plot the test metrics
# from com.wsmtec.machinelearning.utils.plot.plot_evaluation import plot_metrics
# plot_metrics(lr,data_pca,label)

#use the cross validation for evaluation
from com.wsmtec.machinelearning.utils.model_evaluation.cross_validation import cv_score
cv_score(lr,data_pca,label,cv=10)

#use the grid search to find the best param for the data
from com.wsmtec.machinelearning.utils.plot.plot_evaluation import plot_learning_curve,plot_metrics
best_clf = plot_learning_curve(lr,data_pca,label,param_name=['C','intercept_scaling'],params=[[.1,1,10,100],[1,3,5,7]],cv=10)

#plot the result
plot_metrics(best_clf,data_pca,label)