# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

path = 'E:/'

#the first solution
# df = pd.read_excel(path+'/time.xlsx',date_parser=True)
# tmp = np.array(df)

df2 = pd.read_csv(path+'/time.csv',encoding='gb2312')
tmp = np.array(df2)[:,1]
tmp_1 = pd.Series(tmp).map(lambda x:fn(x))
re = np.empty_like(tmp_1)

def fn(x):
    return x.split(' ')

#return the first col is the date col, second is the time col
def fun(df):
    tmp = np.array(df)[:,1]
    def fn(x):
        return x.split(' ')
    tmp_1 = pd.Series(tmp).map(lambda x:fn(x))
    re_date = np.empty_like(tmp_1)
    for i in range(tmp_1.shape[0]):
        re_date[i] = tmp_1[i][0]
    re_time = np.empty_like(tmp_1)
    for j in range(tmp_1.shape[0]):
        re_time[j] = tmp_1[j][1]
    return re_date,re_time