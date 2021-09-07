# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.utils import shuffle

#load the data
path = 'F:\workingData\\201712\\real_time'
# sim_df = pd.read_csv(path+'\\similar_final.csv')
# eval_path = 'F:\workingData\\201802\\real_score_2.0'

df = pd.read_csv(path+'\\applicationInfo_more.csv')
# eval_df = pd.read_csv(eval_path+'/new.csv')
eval_path = 'F:\workingData\\201803\multiModel'
eval_df = pd.read_csv(eval_path+'/data.csv')

cols = ["sfzh","ddbh","shddh","dzr","qx","ts","multi_id_order_n","multi_id_uid_n","multi_id_tel_n",
              "multi_id_bank_n","multi_order_intck_d","multi_loan_same","multi_all_order_intck_d_avg",
              "multi_all_order_intck_d_med","multi_all_loan_mgroup_avg","multi_all_loan_mgroup_median",
              "multi_same_order_intck_d_avg","multi_same_order_intck_d_med","multi_same_loan_mgroup_avg",
              "multi_same_loan_mgroup_median","multi_all_firorder_intck_d","multi_same_firorder_intck_d",
              "multi_id_1m_uid_n","multi_id_1m_order_n","multi_id_1m_order_avg","multi_id_1m_loanm_m",
              "multi_id_1m_loanm_avg","multi_id_1m_loanm_range","multi_id_2m_uid_n","multi_id_2m_order_n",
              "multi_id_2m_order_avg","multi_id_2m_loanm_m","multi_id_2m_loanm_avg","multi_id_2m_loanm_range",
              "multi_id_3m_uid_n","multi_id_3m_order_n","multi_id_3m_order_avg","multi_id_3m_loanm_m",
              "multi_id_3m_loanm_avg","multi_id_3m_loanm_range","multi_id_4m_uid_n","multi_id_4m_order_n",
              "multi_id_4m_order_avg","multi_id_4m_loanm_m","multi_id_4m_loanm_avg","multi_id_4m_loanm_range",
              "multi_id_5m_uid_n","multi_id_5m_order_n","multi_id_5m_order_avg","multi_id_5m_loanm_m",
              "multi_id_5m_loanm_avg","multi_id_5m_loanm_range","multi_id_6m_uid_n","multi_id_6m_order_n",
              "multi_id_6m_order_avg","multi_id_6m_loanm_m","multi_id_6m_loanm_avg","multi_id_6m_loanm_range",
              "multi_id_1m_uid_avg","multi_id_2m_uid_avg","multi_id_3m_uid_avg","multi_id_4m_uid_avg",
              "multi_id_5m_uid_avg","multi_id_6m_uid_avg","multi_yq_order_n","multi_whje","unexpired_amount",
              "multi_yq_money","multi_yq_1m_n","multi_yq_2m_n","multi_yq_3m_n","multi_yq_4m_n","multi_yq_5m_n",
              "multi_yq_6m_n","multi_yq_1m","multi_yq_2m","multi_yq_3m","multi_yq_4m","multi_yq_5m","multi_yq_6m",
              "multi_yq_his","yq_repay_n","yq_repay_wjq_n","multi_yqd_jq_max","multi_yqd_wjq_max","multi_yqd_max",
              "multi_yq_njq_n","multi_yq_jq_n","multi_intct_d_yq","multi_prob_yq_order","re"]
#give the df columns names
df.columns = cols
eval_df.columns = cols


#drop the columns contains no info
#original and drop the cols that means whether or not the person overdue
# df.drop(['sfzh','ddbh','shddh','dzr','ts','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)
# eval_df.drop(['sfzh','ddbh','shddh','dzr','ts','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)
#
# #####  choose the other repay columns to preprossing such as PCA to down the repay info
# repay_df = df.loc[:,['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
#                    'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
#                    'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
#                    'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order']]
# no_repay_df = df.drop(['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
#                    'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
#                    'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
#                    'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order'],axis=1)
#
# #evaluate data to split to repay data and no repay data
# eval_repay_df = eval_df.loc[:,['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
#                    'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
#                    'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
#                    'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order']]
# eval_no_repay_df = eval_df.drop(['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
#                    'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
#                    'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
#                    'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order'],axis=1)
# #fill the range columns nan to 0
# eval_no_repay_df.fillna(0,inplace=True)
#
#
# #get the all satisified data for only leave the ts null cols
# cols_df_repay = repay_df.columns
# sum_of_null= repay_df.isnull().sum()
# sati_cols = sum_of_null<3000
# cols_sati = cols_df_repay[sati_cols]
# repay_df = repay_df[cols_sati]
#
# #evaluate cols
# eval_repay_df = eval_df[cols_sati]
#
# cols_df_repay_no = no_repay_df.columns
# sum_of_null_no = no_repay_df.isnull().sum()
# sati_cols_no = sum_of_null_no < 3000
# cols_sati_no = cols_df_repay_no[sati_cols_no]
# no_repay_df = no_repay_df[cols_sati_no]
#
# #evaluate df of no repay cols
# eval_no_repay_df = eval_no_repay_df[cols_sati_no]
#
# #get the label
# label = np.array(no_repay_df['re'])
# no_repay_df = no_repay_df.drop('re',axis=1)
#
# #eval data and label
# eval_label = np.array(eval_no_repay_df['re'])
# eval_no_repay_df = eval_no_repay_df.drop('re',axis=1)
#
# ##standard the data
# from sklearn.preprocessing import StandardScaler
# repay_scalerer = StandardScaler().fit(repay_df)
# repay_scalered = repay_scalerer.transform(repay_df)
# no_repay_scalerer = StandardScaler().fit(no_repay_df)
# no_repay_scalered = no_repay_scalerer.transform(no_repay_df)
#
# #eval
# eval_repay_scalered = repay_scalerer.transform(eval_repay_df)
# eval_no_repay_scalered = no_repay_scalerer.transform(eval_no_repay_df)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2).fit(repay_scalered)
# repay_new = pca.transform(repay_scalered)
#
# #eval
# eval_repay_new = pca.transform(eval_repay_scalered)
#
# #combine the application and decomposited repay cols to be one
# data = np.concatenate((no_repay_scalered,repay_new),axis=1)
#
# #eval data
# eval_data = np.concatenate((eval_no_repay_scalered,eval_repay_new),axis=1)
#
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l1',C=10,random_state=1234)
#
# #plot the learning-curve
# # skplt.estimators.plot_learning_curve(lr,data,label,cv=10)
#
# #now use the PCA for the final data to decomposition
# data_pca_model = PCA(n_components=20).fit(data)
# data_pca = data_pca_model.transform(data)
# #eval
# eval_data_pca = data_pca_model.transform(eval_data)
#
# #train the model
# lr.fit(data_pca,label)
#
#
# #pred of the eval data
# pred = lr.predict(eval_data_pca)
# prob = lr.predict_proba(eval_data_pca)
#
#
# #plot the result
# skplt.metrics.plot_ks_statistic(eval_label,prob)
# skplt.metrics.plot_roc_curve(eval_label,prob)
# skplt.metrics.plot_precision_recall_curve(eval_label,prob)
# skplt.metrics.plot_confusion_matrix(eval_label,pred)
# plt.show()





# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest = train_test_split(data_pca,label,test_size=.2,random_state=1234)
#
# lr.fit(xtrain,ytrain)
# pred = lr.predict(xtest)
# prob = lr.predict_proba(xtest)
#
# #cross validation for the model
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import make_scorer
# recall_for_cv = make_scorer(metrics.recall_score,greater_is_better=True)
# precision_for_cv = make_scorer(metrics.precision_score,greater_is_better=True)
# f1_for_cv = make_scorer(metrics.f1_score,greater_is_better=True)
# auc_for_cv = make_scorer(metrics.roc_auc_score,greater_is_better=True)
# acc_for_cv = make_scorer(metrics.accuracy_score,greater_is_better=True)
#
# recall_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=recall_for_cv)
# precision_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=precision_for_cv)
# f1_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=f1_for_cv)
# auc_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=auc_for_cv)
# acc_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=acc_for_cv)
# #get the cv result
# print('recall for 10-fold cv = %f'%(recall_cross_validation.mean()))
# print('precision for 10-fold cv = %f'%(precision_cross_validation.mean()))
# print('f1-score for 10-fold cv = %f'%(f1_cross_validation.mean()))
# print('auc for 10-fold cv = %f'%(auc_cross_validation.mean()))
# print('accuracy for 10-fold cv = %f'%(acc_cross_validation.mean()))
#
# skplt.metrics.plot_confusion_matrix(ytest,pred)
# skplt.metrics.plot_roc_curve(ytest,prob)
# skplt.metrics.plot_ks_statistic(ytest,prob)
#
# plt.show()







#use another method to solve it
df_new =df.drop(["sfzh","ddbh","shddh","dzr","ts","qx","multi_yq_1m",
          "multi_yq_2m","multi_yq_3m","multi_yq_4m","multi_yq_5m","multi_yq_6m",
          "multi_intct_d_yq","multi_order_intck_d", "multi_all_order_intck_d_avg",
          "multi_all_order_intck_d_med", "multi_same_order_intck_d_avg","multi_same_order_intck_d_med"],axis=1)
eval_df_new = eval_df.drop(["sfzh","ddbh","shddh","dzr","ts","qx","multi_yq_1m",
          "multi_yq_2m","multi_yq_3m","multi_yq_4m","multi_yq_5m","multi_yq_6m",
          "multi_intct_d_yq","multi_order_intck_d", "multi_all_order_intck_d_avg",
          "multi_all_order_intck_d_med", "multi_same_order_intck_d_avg","multi_same_order_intck_d_med"],axis=1)
#fill nan to 0 for the range
eval_df_new.fillna(0,inplace=True)

df_np = np.array(df_new)
eval_df_np = np.array(eval_df_new)

#org data and label, split the repay and  not repay features
data_org = df_np[:,:df_np.shape[1]-2]
label = df_np[:,-1]
# repay_data = data_org[:,53:]
# no_repay_data = data_org[:,:53]

#eval data and label, same as above
data_new = eval_df_np[:,:eval_df_np.shape[-1]-2]
eval_label = eval_df_np[:,-1]

#standard the all data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data_org)
data_scalered = scaler.transform(data_org)
eval_data_scalered = scaler.transform(data_new)

#get the repay cols after standard to pca
from sklearn.decomposition import PCA
repay_data = data_scalered[:,53:]
no_repay_data = data_scalered[:,:53]
eval_repay_data = eval_data_scalered[:,53:]
eval_no_repay_data = eval_data_scalered[:,:53]

pca_repay = PCA(n_components=2).fit(repay_data)
repay_pca = pca_repay.transform(repay_data)
eval_repay_pca = pca_repay.transform(eval_repay_data)

#combine the decomposited repay cols to the other cols to be one array
data = np.concatenate((no_repay_data,repay_pca),axis=1)
eval_data = np.concatenate((eval_no_repay_data,eval_repay_pca),axis=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.01,penalty='l2').fit(data,label)

#plot the learning curve
# skplt.estimators.plot_learning_curve(lr,data,label,cv=10)
# plt.show()

pred = lr.predict(eval_data)
prob = lr.predict_proba(eval_data)

#plot the prediction result
skplt.metrics.plot_confusion_matrix(eval_label,pred)
skplt.metrics.plot_roc_curve(eval_label,prob)
skplt.metrics.plot_ks_statistic(eval_label,prob)
skplt.metrics.plot_precision_recall_curve(eval_label,prob)
plt.show()

from sklearn import metrics
recall = metrics.recall_score(eval_label,pred)
precision = metrics.precision_score(eval_label,pred)
f1_score = metrics.f1_score(eval_label,pred)
confusin = metrics.confusion_matrix(eval_label,pred)
auc = metrics.roc_auc_score(eval_label,prob[:,1])
fpr,tpr,_ = metrics.roc_curve(eval_label,prob[:,1])
ks = (tpr- fpr).max()

print('Recall = %f  '%recall)
print('Precision = %f  '%precision)
print('F1_score = %f  '%f1_score)
print('AUC = %f  '%auc)
print('KS = %f  '%ks)
print('the Confusion matrix: ')
print(confusin)



