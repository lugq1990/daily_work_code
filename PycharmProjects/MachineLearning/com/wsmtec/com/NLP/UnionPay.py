# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

""" convert the orginal data into a new excel"""
# path = 'F:\workingData\\201804\yidun\yinlian\original_data/'
# df1 = pd.read_csv(path+'/ylzc_info_ckrzhpfnew.csv',encoding='gb18030')
# df2 = pd.read_csv(path+'/ylzc_info_gq1_12new.csv',encoding='gb18030')
# df3 = pd.read_csv(path+'/ylzc_info_j1new.csv',encoding='gb18030')
# df4 = pd.read_csv(path+'/ylzc_info_j2_12new.csv',encoding='gb18030')
# df5 = pd.read_csv(path+'/ylzc_info_jdbnew.csv',encoding='utf-8')
# df6 = pd.read_csv(path+'/ylzc_info_jrnew.csv',encoding='gb18030')
# df7 = pd.read_csv(path+'/ylzc_info_othernew.csv',encoding='utf-8')
#
# out = pd.concat((df1, df2.iloc[:,5:], df3.iloc[:,5:], df4.iloc[:,5:],
#                  df5.iloc[:,5:], df6.iloc[:,5:], df7.iloc[:,5:]), axis=1)
#
# val_list = pd.read_csv(path+'var_list.csv', encoding='gb18030')
# tmp = np.array(np.matrix(val_list).T)
#
# final = np.empty((tmp.shape[0]+ out.shape[0],842))
# final = final.astype(np.str)
#
# for i in range(tmp.shape[1]):
#     for j in range(out.shape[1]):
#         if tmp[0,i] == out.columns[j]:
#             final[:4, i] = tmp[:, i]
#             final[4:, i] = np.array(out[out.columns[j]])
#
#
# df = pd.read_csv(path+'/final.csv', encoding='gb18030').iloc[3:,:]

path = 'F:\workingData\\201804\\unionpay\yinlian\original_data'
df = pd.read_csv(path+'/final_2.csv', encoding='gb18030')
df_object = pd.read_csv(path+'/final.csv', encoding='gb18030')
label = df['Y']
cols = df.columns
names = df_object.iloc[0,:]
it_dict = dict()
for i in range(len(cols)):
    it_dict[cols[i]] = names[i]

""" To find how the each item influences the user whether to overdue"""
# a function to plot the fixed size range count numbers based on the Label
def conver(df, x, range=5, uni_num=15):
    split = True
    if len(np.unique(df)) <= uni_num:
        split = False
    if split and range == 4:
        nums = np.percentile(df,q=[25,50,75]).astype(np.int32)
        if len(nums) <=2:
            nums = np.percentile(df, q=[50,75,90])
        df_new = df.apply(lambda x:'<'+np.str(nums[0]) if x< nums[0] else (
        '>'+np.str(nums[0])+'&&<'+np.str(nums[1]) if x>nums[0] and x<nums[1] else(
            '>'+np.str(nums[1])+'&&'+np.str(nums[2]) if x>nums[1] and x<nums[2] else '>'+np.str(nums[2])
        )))
    if split and range == 5:
        nums = np.percentile(df, q=[20,40,60,80]).astype(np.int32)
        if len(nums) <= 2:
            nums = np.percentile(df, q=[50, 60, 70, 80])
        df_new = df.apply(lambda x:'<'+np.str(nums[0]) if x< nums[0] else (
        '>'+np.str(nums[0])+'&&<'+np.str(nums[1]) if x>nums[0] and x<nums[1] else(
            '>'+np.str(nums[1])+'&&'+np.str(nums[2]) if x>nums[1] and x<nums[2] else(
                '>'+np.str(nums[2])+'&&'+np.str(nums[3]) if x>nums[2] and x<nums[3] else '>'+np.str(nums[3])
            )
        )))
    if not split:
       df_new = df
    plt.figure(figsize=(8, 6))
    sns.countplot(df_new, hue=label)
    plt.xlabel(x)
    # plt.show()

""" Based on the orders number and orders money to find how them to influence the label"""
def vio_plot(orders_key, moneys_key, bins=None, uni_num=10):
    x_data = df[orders_key]
    y_data = df[moneys_key]
    col_index = x_data!=0
    x_data = x_data[col_index]
    y_data = y_data[col_index]
    label_data = label[col_index]
    # this is the categorical columns
    if len(np.unique(x_data)) >= uni_num:
        # if the categorical columns contain much than 10, we use the pd.qcut to split the data
        # to be 10 categorical, if there is same value, combine all of them
        #x_data = pd.qcut(x_data, q=np.arange(0.,1.,0.15).tolist(), duplicates='drop',labels=False)
        if bins is None:
            cut_data = pd.cut(x_data, bins=[0, 5, 10, 15, x_data.max()])
        else: cut_data = pd.cut(x_data, bins=bins)
        x_data = pd.Series([t.right for t in cut_data])
    # plt.figure(figsize=(8,6))
    # sns.violinplot(x_data, y_data, hue=label_data, split=True, cut=0)
    # plt.xlabel(it_dict[orders_key])
    # plt.ylabel(it_dict[moneys_key])

    plt.figure(figsize=(8,6))
    sns.boxplot(x_data, y_data, hue=label_data)
    plt.xlabel(it_dict[orders_key])
    plt.ylabel(it_dict[moneys_key])

""" plot the distribution of the each label based on label column, and just plot the 90% of the datasets"""
def distplot(col, df=df, label_col='Y', rat=95, bins=50):
    df_new = df[[col, label_col]][df[col] != 0]
    max_v = np.percentile(df_new[col], q=[rat])[0]
    data = df_new[[col, label_col]][df_new[col] <= max_v]
    pos_data = data[data[label_col] == 0][col]
    neg_data = data[data[label_col] == 1][col]
    # plot pos and neg in one plot
    fig = plt.figure()
    if label_col == 'Y':
        fig.suptitle(it_dict[col]+'密度估计')
        plt.xlabel(it_dict[col])
    else: fig.suptitle('评分密度估计'), plt.xlabel(np.str(label_col))
    sns.distplot(pos_data, bins=bins, label='normal')
    sns.distplot(neg_data, bins=bins, label='bad')
    plt.legend()
    del df_new
    gc.collect()

""" plot the different month avarage consume money based on label"""
def plot_diff_dist(col1, col2, col3, rat=90, bins=50, is_percent=True, pos_plot=True,
                   max_money=None, ylim=None):
    df_new = df[[col1, col2, col3, 'Y']]
    if max_money is not None:
        is_percent = False
    if is_percent:
        max_v1 = np.percentile(df_new[df_new[col1] != 0][col1], q=[rat])[0]
        max_v2 = np.percentile(df_new[df_new[col2] != 0][col1], q=[rat])[0]
        max_v3 = np.percentile(df_new[df_new[col3] != 0][col1], q=[rat])[0]
        data1 = df_new[[col1, 'Y']][df_new[col1] <= max_v1]
        data2 = df_new[[col2, 'Y']][df_new[col2] <= max_v2]
        data3 = df_new[[col3, 'Y']][df_new[col3] <= max_v3]
    if max_money is not None and is_percent is False:
        data1 = df_new[[col1, 'Y']][df_new[col1] <= max_money]
        data2 = df_new[[col2, 'Y']][df_new[col2] <= max_money]
        data3 = df_new[[col3, 'Y']][df_new[col3] <= max_money]
    pos_1 = data1[data1['Y'] == 0][col1]
    pos_2 = data2[data2['Y'] == 0][col2]
    pos_3 = data3[data3['Y'] == 0][col3]
    neg_1 = data1[data1['Y'] == 1][col1]
    neg_2 = data2[data2['Y'] == 1][col2]
    neg_3 = data3[data3['Y'] == 1][col3]
    # start to plot the data, for neg-data of different month to be in one plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    if pos_plot:
        ax[0].set_title('正常用户密度估计')
        if ylim is not None:
            ax[0].set_ylim((0, ylim))
        sns.distplot(pos_1, bins=bins, label=it_dict[col1], norm_hist=True, ax=ax[0])
        sns.distplot(pos_2, bins=bins, label=it_dict[col2], norm_hist=True, ax=ax[0])
        sns.distplot(pos_3, bins=bins, label=it_dict[col3], norm_hist=True, ax=ax[0])
        ax[0].set_xlabel('消费金额')
        ax[0].legend(loc='upper right')
    # plt.gca().legend((it_dict[col1], it_dict[col2], it_dict[col3]))

    ax[1].set_title('逾期用户密度估计')
    if ylim is not None:
        plt.ylim((0, ylim))
    sns.distplot(neg_1, bins=bins, label=it_dict[col1], norm_hist=True, ax=ax[1])
    sns.distplot(neg_2, bins=bins, label=it_dict[col2], norm_hist=True, ax=ax[1])
    sns.distplot(neg_3, bins=bins, label=it_dict[col3], norm_hist=True, ax=ax[1])
    ax[1].set_xlabel('消费金额')
    ax[1].legend(loc='upper right')
    # plt.gca().legend((it_dict[col1], it_dict[col2], it_dict[col3]))
    del df_new, data1, data2,data3, pos_1,pos_2,pos_3, neg_1,neg_2, neg_3
    gc.collect()

""" Just plot the negative data for 3,6,12 months distplot"""
def plot_dist_neg(col1, col2, col3, rat=90, bins=50, is_percent=True,
                   max_money=None, ylim=None):
    df_new = df[[col1, col2, col3, 'Y']]
    if max_money is not None:
        is_percent = False
    if is_percent:
        max_v1 = np.percentile(df_new[df_new[col1] != 0][col1], q=[rat])[0]
        max_v2 = np.percentile(df_new[df_new[col2] != 0][col1], q=[rat])[0]
        max_v3 = np.percentile(df_new[df_new[col3] != 0][col1], q=[rat])[0]
        data1 = df_new[[col1, 'Y']][df_new[col1] <= max_v1]
        data2 = df_new[[col2, 'Y']][df_new[col2] <= max_v2]
        data3 = df_new[[col3, 'Y']][df_new[col3] <= max_v3]
    if max_money is not None and is_percent is False:
        data1 = df_new[[col1, 'Y']][df_new[col1] <= max_money]
        data2 = df_new[[col2, 'Y']][df_new[col2] <= max_money]
        data3 = df_new[[col3, 'Y']][df_new[col3] <= max_money]
    neg_1 = data1[data1['Y'] == 1][col1]
    neg_2 = data2[data2['Y'] == 1][col2]
    neg_3 = data3[data3['Y'] == 1][col3]

    # start to plot the data, for neg-data of different month to be in one plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_title('逾期用户密度估计')
    if ylim is not None:
        plt.ylim((0, ylim))
    sns.distplot(neg_1, bins=bins, label=it_dict[col1], norm_hist=True, ax=ax)
    sns.distplot(neg_2, bins=bins, label=it_dict[col2], norm_hist=True, ax=ax)
    sns.distplot(neg_3, bins=bins, label=it_dict[col3], norm_hist=True, ax=ax)
    ax.set_xlabel('消费金额')
    ax.legend(loc='upper right')
    # plt.gca().legend((it_dict[col1], it_dict[col2], it_dict[col3]))
    del df_new, data1, data2,data3, neg_1,neg_2, neg_3
    gc.collect()




# 交易金额
pay_moneys_list = ['CDTB068', 'CDTP111', 'CDMC059','CDMC161', 'CDMC133','CDMC179', 'CDMC195', 'CDTC012',
            'CDCT005', 'CDTC002','CDTC034','CDTB003', 'CDTB032','CDTB060']
for it in pay_moneys_list:
    conver(df[it], it_dict[it])

# 对各类消费金额进行密度估计
for it in pay_moneys_list:
    distplot(it)


# 消费笔数
orders_list = ['CDTP126', 'CDMC164','CDMC182','CDMC198','CDTC014','CDCT005',
              'CDTC005','CDTC049','CDTB018','CDTB046','CDTT097','CDTB063']
for it in orders_list:
    conver(df[it], it_dict[it])



# 各类笔数金额与逾期情况
orders_money_dict = {'CDTB093':'CDTB068','CDTP126':'CDTP111',
                     'CDMC164':'CDMC161', 'CDMC136':'CDMC133',
                     'CDMC182':'CDMC179','CDMC198':'CDMC195',
                     'CDTC014':'CDTC012','CDCT002':'CDCT005',
                     'CDTC005':'CDTC002', 'CDTC049':'CDTC034'}
keys = list(orders_money_dict.keys())
for i in range(len(orders_money_dict)):
    vio_plot(keys[i], orders_money_dict[keys[i]])



# 身份特质
values_list = ['CDTB122','CDTB215','CDTB216','CDTB217']
for it in values_list:
    conver(df[it], it_dict[it])

# 银联自身计算各类得分
union_list = ['CSRL003','CSWC001','CSSP003','CSSP002','CSRL004',
               'CSWC002','CSSS003','CSSS001','CSSS002','CSRL001','CSRL002']
for it in union_list:
    conver(df[it], it_dict[it])

# 其他近6个月金额字段
money_new_list = ['CDMC145', 'CDMC079','CDMC191',
                  'CDMC207','CDMC008','CDMC103',
                  'CDMC213','CDMC219','CDMC075',
                  'CDMC248','CDMC112', 'CDMC259',
                  'CDTC008','CDTP045','CDTB155',
                  'CDTT027','CDTB124','CDTB212',
                  'CDMC253']
for it in money_new_list:
    conver(df[it], it_dict[it])

# 对其他金额进行密度估计
for it in money_new_list:
    distplot(it)

# 其他近6个月交易笔数
order_new_list = ['CDMC011','CDMC216','CDMC221','CDMC077','CDMC251',
                  'CDMC115','CDMC262','CDTC010','CDTP059','CDTB157',
                  'CDTT040']
for it in order_new_list:
    conver(df[it], it_dict[it])

# 其他交易笔数与交易金额
new_orders_money_list = {'CDMC148':'CDMC145', 'CDMC082':'CDMC079',
                         'CDMC193':'CDMC191', 'CDMC210':'CDMC207',
                         'CDMC011':'CDMC008', 'CDMC216':'CDMC213',
                         'CDMC221':'CDMC219', 'CDMC077':'CDMC075',
                         'CDMC251':'CDMC248', 'CDMC115':'CDMC112',
                         'CDMC262':'CDMC259', 'CDTB157':'CDTB155'}
keys = list(new_orders_money_list.keys())
for i in range(len(new_orders_money_list)):
    vio_plot(keys[i], new_orders_money_list[keys[i]])


# 对不同月份月均金额分析
diff_list = ['CDTB059', 'CDTB060', 'CDTB061']
plot_diff_dist(diff_list[0], diff_list[1], diff_list[2])



""" this is just a method to compute the score"""
new_path = 'F:\workingData\\201804\yidun\yinlian'
new_df = pd.read_csv(new_path+'/test.csv')
def func(x, is_first=False):
    if x<=2000 and not is_first:
        return 20
    elif x<= 2000 and is_first:
        return 40
    elif x>= 2000 and x<6000:
        return 50
    elif x>=6000 and x<10000:
        return 70
    elif x>=10000 and x<20000:
        return 80
    elif x>20000:
        return 100

df['new_score_1'] = df['col1'].apply(func ).apply(lambda x:x *.6)
df['new_score_2'] = df['col2'].apply(func ).apply(lambda x:x *.2)
df['new_score_3'] = df['col3'].apply(func ).apply(lambda x:x *.2)
df['score'] = np.sum(df[['new_score_1','new_score_2','new_score_3']], axis=1)
df[['col1','col2','col3','score']].to_csv(new_path+'/score.csv', index=False)


