# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'F:\workingData\\201803\loan_data'
org_df = pd.read_csv(path+'/org_data.csv',encoding='gbk')
df = pd.read_csv(path+'/new_data.csv', encoding='gbk')

org_col = ['order_date','time_duration','time_min','mobile','loan_time','activity_day',
           'loan_times_avg','loan_time_max','mingj_min','mingj_max','mingj_last','is_old',
           'wsm_score','tianping','tian_ping_2','zm_score','zm_fake_score','is_in_zm_industry_list',
           'is_in_zm_fake_list','qainsima_re','score_desc','tianping_label','is_machine','this_week',
           'many_loan','group_fake','loss_connect','hit_fake_list','id_fake','indirect_risk_list',
           'identity_diff','risk_level','public_security','overdue_counts','overdue_amount_max',
           'overdue_amount_max_explain','last_over_day','last_over_day_duration']

org_df.columns = org_col

a = np.arange(1,df.shape[1]+1).astype(np.str)
df.columns = a

mobile = df['4']
unique_mobile = np.unique(mobile)

result = np.empty(shape=[unique_mobile.shape[0],org_df.shape[1]]).astype(np.str)

#loop the satisfied data
for i in range(unique_mobile.shape[0]):
    # print(unique_mobile[i])
    # c = unique_mobile[i]
    # re = org_df.loc[lambda df:df.mobile==c,:]
    re = org_df.loc[lambda org_df:org_df.mobile==unique_mobile[i],:]
    if(unique_mobile[i] not in np.array(org_df['mobile'])):
        continue
    result[i,:] = np.array(re).reshape(1,org_df.shape[1])

result = pd.DataFrame(result)
result.columns = org_col

result.to_csv(path+'/result.csv',encoding='gbk',index=False)

