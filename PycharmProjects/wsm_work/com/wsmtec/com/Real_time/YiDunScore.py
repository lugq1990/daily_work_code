# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.externals import joblib

path = 'F:\workingData\\201801'
model_path = 'E:\machineLearningModels\yidun'
df = np.array(pd.read_csv(path+'/data.csv',header=None))
test_df = np.array(pd.read_csv(path+'/test_data.csv'))

id_card = df[:,0]
data = df[:,1:df.shape[1]-2]
label = df[:,df.shape[1]-1].astype(np.int)

#standard the data
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler().fit(data).transform(data)

#accoding the PCA plot prior, decomposition the data to 25
data_pca = PCA(n_components=25).fit(data_scaler).transform(data_scaler)

#split the data to train and test
xtrain,xtest,ytrain,ytest = train_test_split(data_pca,label,test_size=.2,random_state=1234)

lr = LogisticRegression(penalty='l2',C=10,random_state=1234,fit_intercept=True,
                        class_weight={0:.2,1:.8},solver='sag',max_iter=5000)

#fit the model
lr.fit(xtrain,ytrain)
pred = lr.predict(xtest)
prob = lr.predict_proba(xtest)

#plot the learning curve and the metrics
# skplt.estimators.plot_learning_curve(lr,data_pca,label,cv=10)
# skplt.metrics.plot_confusion_matrix(ytest,pred)
# skplt.metrics.plot_roc_curve(ytest,prob)
# skplt.metrics.plot_ks_statistic(ytest,prob)

from com.wsmtec.com.Plot.PlotCoef import plot_coef
from com.wsmtec.com.Plot.PlotEvaluation import plot_learning_curve,plot_metrics
clf = plot_learning_curve(lr,data_pca,label,
                          param_name=['C','intercept_scaling','solver'],
                          params=[[.1,1,10],np.arange(0.001,0.01,.001),['liblinear','lbfgs']])
plot_coef(clf)
plot_metrics(clf,data_pca,label)
print(clf)
lr = clf

#save the best model to disk
joblib.dump(lr,model_path+'/best_lr.pkl')

pred_all = lr.predict_proba(data_pca)
out = pd.DataFrame(pred_all)
out.to_csv(path+"/yidunProb.csv",index=False)

# #use the cross validation to metric the model
# from sklearn.metrics import make_scorer
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score
#
# recall_score = make_scorer(metrics.recall_score,greater_is_better=True)
# precision_score = make_scorer(metrics.precision_score,greater_is_better=True)
# f1_score = make_scorer(metrics.f1_score,greater_is_better=True)
# auc_score = make_scorer(metrics.roc_auc_score,greater_is_better=True)
#
# recall_validation = cross_val_score(lr,data_pca,label,scoring=recall_score)
# precision_validation = cross_val_score(lr,data_pca,label,scoring=precision_score)
# f1_validation = cross_val_score(lr,data_pca,label,scoring=f1_score)
# auc_validation = cross_val_score(lr,data_pca,label,scoring=auc_score)
#
# print('use the 10-fold cross validation for the model, 10-fold recall = ',recall_validation.mean())
# print('use the 10-fold cross validation for the model, 10-fold precision = ',precision_validation.mean())
# print('use the 10-fold cross validation for the model, 10-fold f1_score = ',f1_validation.mean())
# print('use the 10-fold cross validation for the model, 10-fold auc = ',auc_validation.mean())
#
# auc = metrics.roc_auc_score(ytest,prob[:,1])
# recall = metrics.recall_score(ytest,pred)
# precision = metrics.precision_score(ytest,pred)
# fpr,tpr,_ = metrics.roc_curve(ytest,prob[:,1])
# ks = (tpr - fpr).max()
# confusion = metrics.confusion_matrix(ytest,pred)
# f1_score = metrics.f1_score(ytest,pred)
# print('the auc = %f'%auc)
# print('the recall = %f'%recall)
# print('the ks = %f'%ks)
# print('the precision = %f'%precision)
# print('the f1_score = %f'%f1_score)
# print('the confusion matrix = ')
# print(confusion)
#
#
# #use the test data to evaluate the model
# test_data = test_df[:,1:]
# test_label = test_df[:,0]
# #standard the data
# from sklearn.preprocessing import StandardScaler
# test_data_scaler = StandardScaler().fit(test_data).transform(test_data)
#
# #accoding the PCA plot prior, decomposition the data to 25
# test_data_pca = PCA(n_components=25).fit(test_data_scaler).transform(test_data_scaler)
#
# pred = lr.predict(test_data_pca)
# prob = lr.predict_proba(test_data_pca)
#
# auc = metrics.roc_auc_score(test_label,prob[:,1])
# recall = metrics.recall_score(test_label,pred)
# precision = metrics.precision_score(test_label,pred)
# fpr,tpr,_ = metrics.roc_curve(test_label,prob[:,1])
# ks = (tpr - fpr).max()
# confusion = metrics.confusion_matrix(test_label,pred)
# f1_score = metrics.f1_score(test_label,pred)
# print('the auc = %f'%auc)
# print('the recall = %f'%recall)
# print('the ks = %f'%ks)
# print('the precision = %f'%precision)
# print('the f1_score = %f'%f1_score)
# print('the confusion matrix = ')
# print(confusion)
#
# skplt.metrics.plot_confusion_matrix(test_label,pred)
# skplt.metrics.plot_roc_curve(test_label,prob)
# skplt.metrics.plot_ks_statistic(test_label,prob)
# plt.show()



#get the all data prob
prob = lr.predict_proba(data_pca)[:,1]
odds = prob/(1-prob)
score = 600-128*np.log(odds)
out_re = np.concatenate((id_card.reshape(-1,1),score.reshape(-1,1),label.reshape(-1,1)),axis=1)
out_df = pd.DataFrame(out_re)
out_df.columns = ['id_card','score','label']
out_df.to_csv(path+'/score.csv',index=False)








