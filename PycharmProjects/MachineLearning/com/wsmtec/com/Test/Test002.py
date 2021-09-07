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
sati_cols_no = sum_of_null_no<3000
cols_sati_no = cols_df_repay_no[sati_cols_no]
no_repay_df = no_repay_df[cols_sati_no]

print(cols_sati)
print(cols_sati_no)