# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'F:\workingData\\201803\loan_data\\anaData'
# org_df = pd.read_csv(path+'/org_data_2.csv',encoding='gbk')
# df = pd.read_csv(path+'/new_data.csv',encoding='gbk')
df = pd.read_excel(path+'/data.xlsx')
plt.rcParams['font.sans-serif']=['SimHei']

def p():
    plt.show()

#用户的全部逾期天数分析
# over_all_days = np.array(df['over_days'])
# over_all_days = over_all_days[over_all_days!=0]
# sns.countplot(over_all_days)
# plt.show()

#对逾期用户是否命中欺诈清单进行分析
# over_user_fake = df.loc[lambda df:df.over_days>0,['over_days','is_in_zm_fake_list']]
# sns.countplot(over_user_fake.over_days,hue=over_user_fake.is_in_zm_fake_list)
over_user_fake = df.loc[lambda df:df.is_in_zm_fake_list == 'yes',['over_days']]
over_user_fake = np.array(over_user_fake).reshape(-1)
over_user_fake = over_user_fake[over_user_fake>0]
sns.countplot(over_user_fake)
plt.xlabel('逾期天数')
plt.ylabel('逾期人数')
plt.title('命中芝麻欺诈清单逾期分布')
plt.show()

# 逾期用户芝麻分密度估计
# plt.subplot(1,1,1)
# zm_overdue = df.loc[lambda df:df.over_days>0,'wsm_score']
# zm_df = df.loc[lambda df:df.over_days==0,'wsm_score']
# # zm_df = df['mingj_last']
# sns.distplot(zm_df,hist=False,label='正常用户微神马评分密度估计')
# # plt.subplot(1,2,1)
# sns.distplot(zm_overdue,hist=False,label='逾期用户微神马评分密度估计')
# plt.xlabel('分数')
# plt.ylabel('密度')
# plt.title('微神马评分逾期与正常用户密度估计')
# plt.show()

# 第一二期逾期已还及未还 逾期情况分析
# a = [1,1,1,2,3,3,3,1,3,2,4,4,4,4,3,19,4,20,5,20,3,16,3,21,21,21,15,21,8,1,5,11]
# sns.countplot(a)
# plt.xlabel('逾期天数')
# plt.ylabel('逾期人数')
# plt.title('第二期逾期天数')
# p()

