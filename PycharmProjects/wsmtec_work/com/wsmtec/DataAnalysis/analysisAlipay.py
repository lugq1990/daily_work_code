# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def p():
    plt.show()

path = 'F:\workingData\\201803\loan_data'
path_m = 'F:\workingData\\201803\loan_data\medium'
df = pd.read_csv(path+'/result.csv',encoding='gbk')
org_df = pd.read_csv(path+'/new_data.csv',encoding='gbk')

#loan money analysis
# loan_money_df = org_df.loc[lambda df:df.current_loan==1,:]['loan_money']
# sns.countplot(loan_money_df)
# plt.title('loan money')
# p()


#overdue analysis
#第一期逾期人数分析
over_df = org_df.loc[lambda df:df.current_loan==1,:]['first_p_over']
over_array = np.array(over_df).astype(np.int)
# sns.countplot(over_array)
# p()
#对逾期天数超过30天用户情况对应各种分
# over_30_mobile = org_df.loc[lambda df:df.over_days>30,:]['mobile']
# over_30_out = np.empty(shape=[over_30_mobile.shape[0],df.shape[1]]).astype(np.str)
# for i in range(over_30_mobile.shape[0]):
#     if over_30_mobile[i] not in df['mobile']:
#         continue
#     over_30_out[i, :] = np.array(df.loc[lambda df: df.mobile == over_30_mobile[i], :]).reshape(1,df.shape[1])
#
# pd.DataFrame(over_30_out).to_csv(path_m+'/over_30_out.csv',encoding='gbk')

#对全部逾期人数分析
# over_all = over_array[over_array!=0]
# sns.countplot(over_all)
# p()
#第二期逾期情况
# over_second_p = np.array(org_df.loc[lambda df:df.current_loan==2,:]['second_p_over']).astype(np.int)
# sns.countplot(over_second_p[over_second_p!=0])
# p()



#放款用户分数分析
#明镜分
# mingj_score = np.array(df['mingj_last'])
# mingj_score = mingj_score[mingj_score!=0]
# sns.distplot(mingj_score)
# p()
#微神马评分
# mingj_score = np.array(df['wsm_score'])
# mingj_score = mingj_score[mingj_score!=0]
# sns.distplot(mingj_score,bins=30)
# p()
# # 天评分
# mingj_score = np.array(df['tianping'])
# mingj_score = mingj_score[mingj_score!=0]
# #sns.kdeplot(mingj_score,shade=True)
# sns.distplot(mingj_score,bins=40)
# p()
# 天评分2.0
# mingj_score = np.array(df['tian_ping_2'])
# mingj_score = mingj_score[mingj_score!=0]
# #sns.kdeplot(mingj_score,shade=True)
# sns.distplot(mingj_score,bins=40)
# p()
#芝麻分
# mingj_score = np.array(df['zm_score'])
# mingj_score = mingj_score[mingj_score!=0]
# #sns.kdeplot(mingj_score,shade=True)
# sns.distplot(mingj_score,bins=20)
# p()
#芝麻欺诈评分
# mingj_score = np.array(df['zm_fake_score'])
# mingj_score = mingj_score[mingj_score!=0]
# #sns.kdeplot(mingj_score,shade=True)
# sns.distplot(mingj_score,bins=30)
# p()



