# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time

start = time.time()
path = 'F:\workingData\\201709\data\Hive'
df_read = pd.read_csv(path+'/test.txt',sep='\t')

df_read.columns = np.array(pd.Series(np.arange(1,df_read.shape[1] + 1)).map(lambda x:np.str(x)))
df_read.drop(['1','92','93'],axis=1,inplace=True)

data = np.array(df_read.drop(['91'],axis=1))
label = np.array(df_read['91'])

# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.3,random_state=0)

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=1000,penalty='l2',random_state=1234)
# lr.fit(data,label)
# prob = lr.predict_proba(data)[:,1]

#for the label y
# pos_index = (ytest == 0)
# neg_index = (ytest == 1)
# #get the data x
# pos_data = xtest[pos_index]
# neg_data = xtest[neg_index]
# #get all the pos data and neg data probability
# prob_pos = prob[pos_index][:,0]
# prob_neg = prob[neg_index][:,1]
# #make a threshold for pos and neg data label,like 90%
# pos_over_90 = prob_pos[(prob_pos > .9)]
# neg_lower_10 = prob_neg[(prob_neg < .1)]
#
# #get the other data for over 90% pos and lower 10% neg data
# #the index
# other_pos_index = prob_pos[(prob_pos>=.5 & prob_pos<=.9)]
# other_neg_index = (prob_neg <.5 & prob_neg >=.1)
# # pos_lower_90 = ytest[]
# # neg_over_10 = ytest[]
# #only get the not satisified data for pos and neg
# pos_lower_90_data = pos_data[other_pos_index]
# neg_over_10_data = neg_data[other_neg_index]


#the final data and label after throw the not satisified data
# data_return = np.empty((-1,data.shape[1]))
# label_return = np.empty((-1))


#use the lightGBM
# import lightgbm as lgb
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': ['auc', 'binary_error'],
#     'num_leaves': 101,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.9,
#     'bagging_freq': 5,
#     'verbose': 0,
# }

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000,penalty='l2',random_state=1234)

def iter_lr(data,label,iter=10,uper = .7,lower = .3):
    # for i in range(iter):
    i = 0
    return_data = np.empty((1,data.shape[1]))
    return_label = np.empty((1))
    while(True):
        print('iter num is  %d',i+1)
        time_s = time.time()
        data_part = data
        label_part = label
        # lgb_train = lgb.Dataset(data_part,label_part)
        # lgb_test = lgb.Dataset(data_part,label_part)
        # gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_test,early_stopping_rounds=5)
        lr.fit(data_part,label_part)
        prob_part = lr.predict_proba(data_part)[:,1]
        prob_part_pd = pd.Series(prob_part)
        #get the satisified data and label
        data_left = data[(prob_part_pd < uper)&(prob_part_pd > lower)]
        label_left = label[(prob_part_pd < uper)&(prob_part_pd > lower)]
        #return the other data
        other_data = data[(prob_part_pd > uper) | (prob_part_pd < lower)]
        other_label = label[(prob_part_pd > uper) | (prob_part_pd < lower)]
        #insure the data to be all
        return_data = np.concatenate((return_data,other_data),axis=0)
        return_label = np.concatenate((return_label,other_label),axis=0)
        print("the iter is ",i+1," and the data shape is ",data_left.shape," and the label is ",label_left.shape)
        print("the iter is ", i + 1, " and the other data shape is ", return_data.shape, " and the other label is ", return_label.shape)
        print("one iter use time ",time.time() - time_s)
        if((data.shape[0] < 100000) | ((data.shape[0] - data_left.shape[0]) < 100)):
            break
        #change the data and label
        data = data_left
        label = label_left
        i += 1
    return_data = np.delete(return_data,0,axis=0)
    return_label = np.delete(return_label,0,axis=0)
    return data_left,label_left,return_data,return_label

data_return,label_return,other_data,other_label = iter_lr(data=data,label=label)

print('data_return shape',data_return.shape)
print('label_return is ',label_return)
cols = ['installment_days','loan_mgroup','gender','age','id_order_n',
               'id_uid_n','id_tel_n','id_bank_n','order_intck_d','loan_same',
               'all_order_intck_d_avg','all_order_intck_d_med','all_loan_mgroup_avg',
               'all_loan_mgroup_median','same_order_intck_d_avg','same_order_intck_d_med',
               'same_loan_mgroup_avg','same_loan_mgroup_median','all_firorder_intck_d',
               'same_firorder_intck_d','id_1m_uid_n','id_1m_order_n','id_1m_order_avg','id_1m_loanm_m',
               'id_1m_loanm_avg','id_1m_loanm_range','id_2m_uid_n','id_2m_order_n','id_2m_order_avg',
               'id_2m_loanm_m','id_2m_loanm_avg','id_2m_loanm_range','id_3m_uid_n','id_3m_order_n',
               'id_3m_order_avg','id_3m_loanm_m','id_3m_loanm_avg','id_3m_loanm_range','id_4m_uid_n',
               'id_4m_order_n','id_4m_order_avg','id_4m_loanm_m','id_4m_loanm_avg','id_4m_loanm_range',
               'id_5m_uid_n','id_5m_order_n','id_5m_order_avg','id_5m_loanm_m','id_5m_loanm_avg','id_5m_loanm_range',
               'id_6m_uid_n','id_6m_order_n','id_6m_order_avg','id_6m_loanm_m','id_6m_loanm_avg','id_6m_loanm_range',
               'id_1m_uid_avg','id_2m_uid_avg','id_3m_uid_avg','id_4m_uid_avg','id_5m_uid_avg','id_6m_uid_avg',
               'relat_id_same','pro_qy_id_same','pro_qy_tel_same','pro_tel_id_same','pro_rt_qy_same',
               'pro_rt_id_same','pro_rt_tel_same','ct_qy_id_ same','ct_qy_tel_same','ct_tel_id_same',
               'ct_rt_qy_same','ct_rt_id_same','ct_rt_tel_same','id_6m_qy_n','id_5m_qy_n','id_4m_qy_n',
               'id_3m_qy_n','id_2m_qy_n','id_1m_qy_n','id_cy_bin','id_pr_bin','qy_cy_bin','qy_pr_bin',
               'tel_cy_bin','tel_pr_bin','rela_cy_bin','rela_pr_bin','result']
path = 'F:\workingData\\201711\\NotSeperatableData'
out = np.concatenate((data_return,label_return.reshape(-1,1)),axis=1)
out_pd = pd.DataFrame(out)
out_pd.columns = cols
out_pd.to_csv(path+'/out_data.csv',index=False)
#make the other data for easy seperate data
out_other = np.concatenate((other_data,other_label.reshape(-1,1)),axis=1)
out_other = pd.DataFrame(out_other)
out_other.columns = cols
out_other.to_csv(path+'/easy_seperate_data.csv',index=False)
end = time.time()
print('the time cost of the for loop is ',end-start)

