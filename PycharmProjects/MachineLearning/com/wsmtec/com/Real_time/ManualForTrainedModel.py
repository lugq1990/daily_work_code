# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt

org_path = 'F:\workingData\\201801'
path = 'F:\workingData\\201804\yidun'
model_path = 'E:\machineLearningModels\yidun'
df = np.array(pd.read_csv(org_path+'/data.csv',header=None))
test_df = np.array(pd.read_csv(path+'/yidun_liveness.csv', header=None))

# print('show ',pd.read_csv(org_path+'/data.csv',header=None).head())
# print('org show', pd.read_csv(path+'/app_liveness_yidun_20180412.csv', header=None).head())

id_card = df[:,0]
data = df[:,1:df.shape[1]-1]
label = df[:,df.shape[1]-1].astype(np.int)

pred_id_and_other_cols = test_df[:,0:5]
pred_data = test_df[:,5:-2]

# standard the data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(data)
# data_scaler = scaler.transform(data)
# # save the scaler model to disk
# joblib.dump(scaler, model_path+'/scaler.pkl')
#
# #accoding the PCA plot prior, decomposition the data to 25
# pca = PCA(n_components=25).fit(data_scaler)
# data_pca = pca.transform(data_scaler)
# # save the pca model to disk
# joblib.dump(pca, model_path+'/pca.pkl')
#
# #split the data to train and test
# xtrain,xtest,ytrain,ytest = train_test_split(data_pca,label,test_size=.2,random_state=1234)
#
# lr = LogisticRegression(penalty='l2',C=10,random_state=1234,fit_intercept=True,
#                         class_weight={0:.2,1:.8},solver='sag',max_iter=5000)
#
# #fit the model
# lr.fit(xtrain,ytrain)
# pred = lr.predict(xtest)
# prob = lr.predict_proba(xtest)
#
# # plot the learning curve and the metrics
# skplt.estimators.plot_learning_curve(lr,data_pca,label,cv=10)
# skplt.metrics.plot_confusion_matrix(ytest,pred)
# skplt.metrics.plot_roc_curve(ytest,prob)
# skplt.metrics.plot_ks_statistic(ytest,prob)
# plt.show()
#
# from com.wsmtec.com.Plot.PlotCoef import plot_coef
# from com.wsmtec.com.Plot.PlotEvaluation import plot_learning_curve,plot_metrics
# clf = plot_learning_curve(lr,data_pca,label,
#                           param_name=['C','intercept_scaling','solver'],
#                           params=[[.1,1,10],np.arange(0.001,0.01,.001),['liblinear','lbfgs']])
# plot_coef(clf)
# plot_metrics(clf,data_pca,label)
# print(clf)
# lr = clf

#save the best model to disk
#joblib.dump(lr,model_path+'/best_lr.pkl')
# lr = joblib.load(model_path+'/best_lr.pkl')
#
# pred_all = lr.predict_proba(data_pca)
# out = pd.DataFrame(pred_all)
# out.to_csv(path+"/yidunProb.csv",index=False)



# Get the all model from disk
scaler_model = joblib.load(model_path+'/scaler.pkl')
pca_model = joblib.load(model_path+'/pca.pkl')
lr_model = joblib.load(model_path+'/best_lr.pkl')

pred_scaled = scaler_model.transform(pred_data)
pred_pca = pca_model.transform(pred_scaled)
prob = lr_model.predict_proba(pred_pca)[:,1]

odds = prob/(1-prob)
score = 600 - 128*np.log(odds)

out = np.concatenate((pred_id_and_other_cols, score.reshape(-1,1)),axis=1)
out_df = pd.DataFrame(out,columns=['id','mobile','orderId','req_id','biz_id','score'])

out_df.to_csv(path + '/yidun_result_score.csv', index=False)
