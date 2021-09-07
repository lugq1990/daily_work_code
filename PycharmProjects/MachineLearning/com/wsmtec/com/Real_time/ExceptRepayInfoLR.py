# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.utils import shuffle

#load the data
path = 'F:\workingData\\201712\\real_time'
# df = pd.read_csv(path+'\\final_data_1219.csv')
# df = pd.read_csv(path+'\\originalData.csv')
sim_df = pd.read_csv(path+'\\similar_final.csv')
# new_df = pd.read_csv(path + '\\new_data.csv')

df = pd.read_csv(path+'\\applicationInfo_more.csv')
# df = pd.read_csv(path+'\\original_without_revise.csv')


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


#drop the columns contains no info
#original and drop the cols that means whether or not the person overdue
df.drop(['sfzh','ddbh','shddh','dzr','ts','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)


#####  choose the other repay columns to preprossing such as PCA to down the repay info
repay_df = df.loc[:,['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
                   'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
                   'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
                   'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order']]
no_repay_df = df.drop(['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
                   'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
                   'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
                   'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order'],axis=1)


#get the all satisified data for only leave the ts null cols
cols_df_repay = repay_df.columns
sum_of_null= repay_df.isnull().sum()
sati_cols = sum_of_null<3000
cols_sati = cols_df_repay[sati_cols]
repay_df = repay_df[cols_sati]

cols_df_repay_no = no_repay_df.columns
sum_of_null_no = no_repay_df.isnull().sum()
sati_cols_no = sum_of_null_no < 3000
cols_sati_no = cols_df_repay_no[sati_cols_no]
no_repay_df = no_repay_df[cols_sati_no]

#get the label
label = np.array(no_repay_df['re'])
no_repay_df = no_repay_df.drop('re',axis=1)
#from com.wsmtec.com.Plot.PlotPCA import plot_pca
#plot_pca(repay_df)

#impute the NaN data
# from sklearn.preprocessing import Imputer
# repay_df = Imputer(strategy='mean').fit(np.array(repay_df)).transform(np.array(repay_df))
# no_repay_df = Imputer(strategy='mean').fit(np.array(no_repay_df)).transform(np.array(no_repay_df))

##standard the data
from sklearn.preprocessing import StandardScaler
repay_scalered = StandardScaler().fit(repay_df).transform(repay_df)
no_repay_scalered = StandardScaler().fit(no_repay_df).transform(no_repay_df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(repay_scalered)
repay_new = pca.transform(repay_scalered)

#combine the application and decomposited repay cols to be one
data = np.concatenate((no_repay_scalered,repay_new),axis=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',C=10,random_state=1234)

#plot the learning-curve
skplt.estimators.plot_learning_curve(lr,data,label,cv=10)

#now use the PCA for the final data to decomposition
data_pca = PCA(n_components=20).fit(data).transform(data)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data_pca,label,test_size=.2,random_state=1234)

lr.fit(xtrain,ytrain)
pred = lr.predict(xtest)
prob = lr.predict_proba(xtest)

#cross validation for the model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
recall_for_cv = make_scorer(metrics.recall_score,greater_is_better=True)
precision_for_cv = make_scorer(metrics.precision_score,greater_is_better=True)
f1_for_cv = make_scorer(metrics.f1_score,greater_is_better=True)
auc_for_cv = make_scorer(metrics.roc_auc_score,greater_is_better=True)
acc_for_cv = make_scorer(metrics.accuracy_score,greater_is_better=True)

recall_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=recall_for_cv)
precision_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=precision_for_cv)
f1_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=f1_for_cv)
auc_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=auc_for_cv)
acc_cross_validation = cross_val_score(lr,data,label,cv=10,scoring=acc_for_cv)
#get the cv result
print('recall for 10-fold cv = %f'%(recall_cross_validation.mean()))
print('precision for 10-fold cv = %f'%(precision_cross_validation.mean()))
print('f1-score for 10-fold cv = %f'%(f1_cross_validation.mean()))
print('auc for 10-fold cv = %f'%(auc_cross_validation.mean()))
print('accuracy for 10-fold cv = %f'%(acc_cross_validation.mean()))

skplt.metrics.plot_confusion_matrix(ytest,pred)
skplt.metrics.plot_roc_curve(ytest,prob)
skplt.metrics.plot_ks_statistic(ytest,prob)





#use the train algotrithm
sim_df.columns = cols

#drop the columns contains no info
#original and drop the cols that means whether or not the person overdue
sim_df.drop(['sfzh','ddbh','shddh','dzr','ts','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)


#####  choose the other repay columns to preprossing such as PCA to down the repay info
repay_sim_df = sim_df.loc[:,['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
                   'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
                   'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
                   'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order']]
no_repay_sim_df = sim_df.drop(['multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n',
                   'multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_his','yq_repay_n',
                   'yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n',
                   'multi_yq_jq_n','multi_intct_d_yq','multi_prob_yq_order'],axis=1)


#get the all satisified data for only leave the ts null cols
cols_df_repay = repay_sim_df.columns
sum_of_null= repay_sim_df.isnull().sum()
sati_cols = sum_of_null<3000
cols_sati = cols_df_repay[sati_cols]
repay_df = repay_sim_df[cols_sati]

cols_df_repay = no_repay_sim_df.columns
sum_of_null= no_repay_sim_df.isnull().sum()
sati_cols = sum_of_null<3000
cols_sati = cols_df_repay[sati_cols]
no_repay_sim_df = no_repay_sim_df[cols_sati]

#get the label
label_sim = np.array(no_repay_sim_df['re'])
repay_sim_df = no_repay_sim_df.drop('re',axis=1)
#from com.wsmtec.com.Plot.PlotPCA import plot_pca
#plot_pca(repay_df)

from sklearn.preprocessing import Imputer
repay_df = Imputer(strategy='mean').fit(np.array(repay_sim_df)).transform(np.array(repay_sim_df))
no_repay_df = Imputer(strategy='mean').fit(np.array(no_repay_sim_df)).transform(np.array(no_repay_sim_df))

##standard the data
from sklearn.preprocessing import StandardScaler
repay_scalered = StandardScaler().fit(repay_sim_df).transform(repay_sim_df)
no_repay_scalered = StandardScaler().fit(repay_sim_df).transform(repay_sim_df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(repay_scalered)
repay_new = pca.transform(repay_scalered)

#combine the application and decomposited repay cols to be one
data_sim = np.concatenate((no_repay_scalered,repay_new),axis=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',C=10,random_state=1234)

#plot the learning-curve
# skplt.estimators.plot_learning_curve(lr,data_sim,label,cv=10)

#now use the PCA for the final data to decomposition
data_sim_pca = PCA(n_components=20).fit(data_sim).transform(data_sim)

from com.wsmtec.com.Real_time.LrForUnionPay import recusive_loop
data_min_max,label_np ,_,_ = recusive_loop(data,label,data_sim_pca,label_sim,thre=.9,pos_ratio=4)

from sklearn.utils import shuffle
data_min_max,label_np = shuffle(data_min_max,label_np)

#split the data to train and test
xtrain,xtest,ytrain,ytest = train_test_split(data_min_max,label_np,test_size=.2,random_state=1234)
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)
prob = model.predict_proba(xtest)
recall_cross_validation = cross_val_score(model,data_min_max,label_np,cv=10,scoring=recall_for_cv)
precision_cross_validation = cross_val_score(model,data_min_max,label_np,cv=10,scoring=precision_for_cv)
f1_cross_validation = cross_val_score(model,data_min_max,label_np,cv=10,scoring=f1_for_cv)
auc_cross_validation = cross_val_score(model,data_min_max,label_np,cv=10,scoring=auc_for_cv)
acc_cross_validation = cross_val_score(model,data_min_max,label_np,cv=10,scoring=acc_for_cv)
#get the cv result
print('recall for 10-fold cv = %f'%(recall_cross_validation.mean()))
print('precision for 10-fold cv = %f'%(precision_cross_validation.mean()))
print('f1-score for 10-fold cv = %f'%(f1_cross_validation.mean()))
print('auc for 10-fold cv = %f'%(auc_cross_validation.mean()))
print('accuracy for 10-fold cv = %f'%(acc_cross_validation.mean()))



