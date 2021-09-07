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
df = pd.read_csv(path+'\\originalData.csv')
sim_df = pd.read_csv(path+'\\similar_final.csv')
new_df = pd.read_csv(path + '\\new_data.csv')

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
sim_df.columns = cols
new_df.columns = cols

#get the all satisified data for only leave the ts null cols
cols_df = df.columns
sum_of_null= df.isnull().sum()
sati_cols = sum_of_null<2000
cols_sati = cols_df[sati_cols]
df_sati = df[cols_sati]

#add the other data for test,also drop the columns that same to the train data
df_test = sim_df[cols_sati]
df_new = new_df[cols_sati]

#drop the columns contains no info
#original and drop the cols that means whether or not the person overdue
df_sati.drop(['sfzh','ddbh','shddh','dzr','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)
df_test.drop(['sfzh','ddbh','shddh','dzr','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)
df_new.drop(['sfzh','ddbh','shddh','dzr','qx','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m'],axis=1,inplace=True)
# df_sati.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yqd_max','multi_yqd_jq_max','multi_yq_his','yq_repay_wjq_n','yq_repay_n','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_njq_n','multi_yq_jq_n','multi_yqd_wjq_max'],axis=1,inplace=True)
# df_test.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yqd_max','multi_yqd_jq_max','multi_yq_his','yq_repay_wjq_n','yq_repay_n','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_njq_n','multi_yq_jq_n','multi_yqd_wjq_max'],axis=1,inplace=True)
# df_new.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yqd_max','multi_yqd_jq_max','multi_yq_his','yq_repay_wjq_n','yq_repay_n','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_njq_n','multi_yq_jq_n','multi_yqd_wjq_max'],axis=1,inplace=True)
# df_sati.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_his','yq_repay_n','yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n','multi_yq_jq_n','multi_prob_yq_order'],axis=1,inplace=True)
# df_test.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_his','yq_repay_n','yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n','multi_yq_jq_n','multi_prob_yq_order'],axis=1,inplace=True)
# df_new.drop(['sfzh','ddbh','shddh','dzr','qx','ts','multi_yq_order_n','multi_whje','unexpired_amount','multi_yq_money','multi_yq_1m_n','multi_yq_2m_n','multi_yq_3m_n','multi_yq_4m_n','multi_yq_5m_n','multi_yq_6m_n','multi_yq_1m','multi_yq_2m','multi_yq_3m','multi_yq_4m','multi_yq_5m','multi_yq_6m','multi_yq_his','yq_repay_n','yq_repay_wjq_n','multi_yqd_jq_max','multi_yqd_wjq_max','multi_yqd_max','multi_yq_njq_n','multi_yq_jq_n','multi_prob_yq_order'],axis=1,inplace=True)


#compute the columns that contains zero
cols_contine_zeros = (df_sati==0)
count_zeros_cols = cols_contine_zeros.sum()

#compute the cols that contains much more 0
cols_for_zero = (np.array(count_zeros_cols)>1000)
col_num_more_zero = np.sum(cols_for_zero)
col_num = df_sati.shape[1]
col_ratio = col_num_more_zero/col_num
print('the cols contain so much zero num ',col_num_more_zero)
print('the all col num ',col_num)
print('the col contain more zero cols ratio for all cols ',col_ratio)

#get the data and label
data = df_sati.drop('re',axis=1)
label = df_sati['re']
data_np = np.array(data)
label_np = np.array(label)
#get the test data and label
test_data_np = np.array(df_test.drop('re',axis=1))
test_label_np = np.array(df_test['re'])
####add the new data to be splited into data and label
new_data_np = np.array(df_new.drop('re',axis=1))
new_label_np = np.array(df_new['re'])

#compute the pos and neg data ratio
pos_neg_ratio = np.sum(label_np==1)/np.sum(label_np==0)
print('the neg data divided by pos data rotio = ',pos_neg_ratio)

#show the data for leaved ,now the only null col is the ts
#use the sklearn imputer algorithm to make the null data to no-null
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(data_np)
data_np = imp.transform(data_np)
#also for the test data to make the NaN cols to be no-Null
imp.fit(test_data_np)
test_data_np = imp.transform(test_data_np)
###compute the new data imputer
imp.fit(new_data_np)
new_data_np = imp.transform(new_data_np)

#before the logistic regression ,make the data for min-max
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(data_np)
data_min_max = scaler.transform(data_np)
#scale the data for the test and
scaler_test = MinMaxScaler().fit(test_data_np)
test_data_min_max = scaler_test.transform(test_data_np)
####added the new data
scaler_new  = MinMaxScaler().fit(new_data_np)
new_data_min_max = scaler_new.transform(new_data_np)

#use the PCA to decomposition
from sklearn.decomposition import PCA
pca = PCA(n_components=15).fit(data_min_max)
data_pca = pca.transform(data_min_max)
#just for the simplity
data_min_max = data_pca
#decomposite the test data
pca_test = PCA(n_components=15).fit(test_data_min_max)
test_data_pca = pca_test.transform(test_data_min_max)
test_data_min_max = test_data_pca
####added the new data
pca_new = PCA(n_components=15).fit(new_data_min_max)
new_data_pca = pca_new.transform(new_data_min_max)
new_data_min_max = new_data_pca

#split the data to train and test
#### now that we have decide the param for the model, just use the all train data to fit the lr

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data_min_max,label_np,test_size = .2,random_state=1234)

#make the lr model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000,C=100,penalty='l1',random_state=1234)
lr.fit(xtrain,ytrain)

#save the model
lr.fit(data_min_max,label_np)
joblib.dump(lr,path+"//lr_original.pkl")

#plot the original learning-curve
skplt.estimators.plot_learning_curve(lr,data_min_max,label_np,cv=10)
plt.title("original plot")

#get the prob and pred
prob = lr.predict_proba(xtest)[:,1]
pred = lr.predict(xtest)
score = lr.score(xtest,ytest)

#evaluate the model
from sklearn import metrics
auc = metrics.roc_auc_score(ytest,prob)
fpr,tpr,_ = metrics.roc_curve(ytest,prob)
ks = (tpr - fpr).max()
recall = metrics.recall_score(ytest,pred)
precision = metrics.precision_score(ytest,pred)
confusion = metrics.confusion_matrix(ytest,pred)
f1 = metrics.f1_score(ytest,pred)
print('auc = ',auc)
print('ks= ',ks)
print('recall=',recall)
print('precision=',precision)
print('confusion=',confusion)
print('f1-score=',f1)

#plot the roc and confusion matrix
import matplotlib.pyplot as plt
import scikitplot as skplt
# prob_plot = lr.predict_proba(xtest)
# skplt.metrics.plot_roc_curve(ytest,prob_plot)
# skplt.metrics.plot_confusion_matrix(ytest,pred)
# plt.show()



#make the cross validation for the lr
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
recall_for_cv = make_scorer(metrics.recall_score,greater_is_better=True)
precision_for_cv = make_scorer(metrics.precision_score,greater_is_better=True)
f1_for_cv = make_scorer(metrics.f1_score,greater_is_better=True)
auc_for_cv = make_scorer(metrics.roc_auc_score,greater_is_better=True)
acc_for_cv = make_scorer(metrics.accuracy_score,greater_is_better=True)

#compute the evaluation for the model
#use the 10-fold cross validation
recall_cross_validation = cross_val_score(lr,data_min_max,label_np,cv=10,scoring=recall_for_cv)
precision_cross_validation = cross_val_score(lr,data_min_max,label_np,cv=10,scoring=precision_for_cv)
f1_cross_validation = cross_val_score(lr,data_min_max,label_np,cv=10,scoring=f1_for_cv)
auc_cross_validation = cross_val_score(lr,data_min_max,label_np,cv=10,scoring=auc_for_cv)
acc_cross_validation = cross_val_score(lr,data_min_max,label_np,cv=10,scoring=acc_for_cv)
#get the cv result
print('recall for 10-fold cv = %f'%(recall_cross_validation.mean()))
print('precision for 10-fold cv = %f'%(precision_cross_validation.mean()))
print('f1-score for 10-fold cv = %f'%(f1_cross_validation.mean()))
print('auc for 10-fold cv = %f'%(auc_cross_validation.mean()))
print('accuracy for 10-fold cv = %f'%(acc_cross_validation.mean()))



#above is all the data to be featured for train and test,
# now add the recusive loop for choosing the similar data


def recusive_loop(trained_data=data_min_max,trained_label=label_np,
                  test_data=test_data_min_max,test_label=test_label_np,
                  thre=.9,pos_ratio=1,iters=15,max_nums=50000):
    data_min_max = trained_data
    label_np = trained_label
    test_data_min_max = test_data
    test_label_np = test_label
    #lr = estimator
    miss_return_data = np.empty((1, data_min_max.shape[1]))
    miss_return_label = np.empty((1))
    sati_return_data = np.empty((1, data_min_max.shape[1]))
    sati_return_label = np.empty((1))
    i = 0
    while (True):
        print("*****************the next iter ****************")
        print("start train the model step %d"%(i+1))
        print("the data shape is ", data_min_max.shape)
        # build the model
        lr.fit(data_min_max, label_np)
        # predict the prob and pred for the test data
        pred = lr.predict(test_data_min_max)
        prob = lr.predict_proba(test_data_min_max)

        # get the correct and miss pred index
        correct_pred = (test_label_np == pred)
        miss_pred = (test_label_np != pred)
        # get the correct data prob(for the next step to depart the threshold for correct pred data) and miss data
        correct_prob = prob[correct_pred]
        correct_data = test_data_min_max[correct_pred]
        correct_label = test_label_np[correct_pred]
        # get the correct pred prob for pos and neg data depart
        correct_pos_prob = correct_prob[correct_label == 1][:,1]
        correct_neg_prob = correct_prob[correct_label == 0][:,0]
        correct_pos_data = correct_data[correct_label == 1]
        correct_neg_data = correct_data[correct_label == 0]
        correct_pos_label = correct_label[correct_label == 1]
        correct_neg_label = correct_label[correct_label == 0]
        # get the satisified data and not satisified data by the threshold
        sati_pos_data = correct_pos_data[correct_pos_prob > thre]
        sati_neg_data = correct_neg_data[correct_neg_prob > thre]
        not_sati_pos_data = correct_pos_data[correct_pos_prob <= thre]
        not_sati_neg_data = correct_neg_data[correct_neg_prob <= thre]
        # get the satisified ant not sati label
        sati_pos_label = correct_pos_label[correct_pos_prob > thre]
        sati_neg_label = correct_neg_label[correct_neg_prob > thre]
        not_sati_pos_label = correct_pos_label[correct_pos_prob <= thre]
        not_sati_neg_label = correct_neg_label[correct_neg_prob <= thre]
        # get the miss pred data
        miss_data = test_data_min_max[miss_pred]
        miss_label = test_label_np[miss_pred]

        # now combine the all sati and not sati data to be one dataset
        # all_sati_data = np.concatenate((data_min_max, sati_pos_data, sati_neg_data), axis=0)
        # all_sati_label = np.concatenate((label_np, sati_pos_label, sati_neg_label), axis=0)
        ##change the sati data(not included the pos data because pos data is so many,
        ##just not want to change the original data prob distribution )

        ####for now I will random add the neg data(that is the 0) to the dataset
        sati_neg_data,sati_neg_label = shuffle(sati_neg_data,sati_neg_label)
        random_add_data = sati_neg_data[:sati_pos_data.shape[0]*pos_ratio,:]
        random_add_label = sati_neg_label[:sati_pos_data.shape[0]*pos_ratio]
        #the neg data leaved
        sati_neg_data = sati_neg_data[sati_pos_data.shape[0]*pos_ratio:,:]
        sati_neg_label = sati_neg_label[sati_pos_data.shape[0]*pos_ratio:]
        #now get the all pos data and added random_add_data
        all_sati_data = np.concatenate((data_min_max, sati_pos_data,random_add_data), axis=0)
        all_sati_label = np.concatenate((label_np, sati_pos_label,random_add_label), axis=0)

        #return the miss data and label,
        #add the postive data to return(that means training pos data will not changed)
        miss_return_data = np.concatenate((miss_data,not_sati_neg_data, sati_neg_data,not_sati_pos_data), axis=0)
        miss_return_label = np.concatenate((miss_label,not_sati_neg_label,sati_neg_label, not_sati_pos_label), axis=0)

        # now that we get all the satisified and not satisified data and label ,we can train the model again
        # just make a little simplified transformation ,make the train data to be data_min_max
        data_min_max = all_sati_data
        label_np = all_sati_label
        test_data_min_max = miss_return_data
        test_label_np = miss_return_label

        print("finished the step  %d"%(i+1))
        i += 1
        if (i > iters or data_min_max.shape[0] > max_nums):
            #save the model
            joblib.dump(lr,path+'//lr.pkl')
            sati_return_data = np.delete(data_min_max,0,axis=0)
            sati_return_label = np.delete(label_np,0,axis=0)
            miss_return_data = np.delete(miss_return_data,0,axis=0)
            miss_return_label = np.delete(miss_return_label,0,axis=0)
            return sati_return_data,sati_return_label,miss_return_data,miss_return_label
            break


data_min_max,label_np ,_,_ = recusive_loop(data_min_max,label_np,test_data_min_max,test_label_np,thre=.9,pos_ratio=4)
#add the shuffle for the data
from sklearn.utils import shuffle
data_min_max,label_np = shuffle(data_min_max,label_np)

print("the new data shape is ",data_min_max.shape)
model = joblib.load(path+'//lr.pkl')

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
# skplt.estimators.plot_learning_curve(model,data_min_max,label_np,cv=10)
# plt.title("the new model learning curve")

#plot the ks,recall,roc curve
skplt.metrics.plot_confusion_matrix(ytest,pred)
skplt.metrics.plot_ks_statistic(ytest,prob)
skplt.metrics.plot_roc_curve(ytest,prob)
plt.show()


# print("********************************************************")
# print("the added new data!!!")
# model = joblib.load(path+"//lr.pkl")
# pred_new = model.predict(new_data_min_max)
# prob_new = model.predict_proba(new_data_min_max)
# recall_cross_validation = cross_val_score(model,new_data_min_max,new_label_np,cv=10,scoring=recall_for_cv)
# precision_cross_validation = cross_val_score(model,new_data_min_max,new_label_np,cv=10,scoring=precision_for_cv)
# f1_cross_validation = cross_val_score(model,new_data_min_max,new_label_np,cv=10,scoring=f1_for_cv)
# auc_cross_validation = cross_val_score(model,new_data_min_max,new_label_np,cv=10,scoring=auc_for_cv)
# acc_cross_validation = cross_val_score(model,new_data_min_max,new_label_np,cv=10,scoring=acc_for_cv)
# print('recall for 10-fold cv = %f'%(recall_cross_validation.mean()))
# print('precision for 10-fold cv = %f'%(precision_cross_validation.mean()))
# print('f1-score for 10-fold cv = %f'%(f1_cross_validation.mean()))
# print('auc for 10-fold cv = %f'%(auc_cross_validation.mean()))
# print('accuracy for 10-fold cv = %f'%(acc_cross_validation.mean()))
#
# skplt.metrics.plot_roc_curve(new_label_np,prob_new)
# plt.title("the new data roc curve")
# skplt.metrics.plot_ks_statistic(new_label_np,prob_new)
# plt.title('the new data ks curve')
# skplt.metrics.plot_confusion_matrix(new_label_np,pred_new)
# plt.title('the new data confusion matrix')
# plt.show()
#
# id_col = np.array(new_df['sfzh'])
# out = np.concatenate((id_col,prob_new),axis=1)