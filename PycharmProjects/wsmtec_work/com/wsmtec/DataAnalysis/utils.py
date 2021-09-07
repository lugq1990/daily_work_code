# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

def overdue_produce_based_on_score(path,file_name,split_range=50,min=100,max=900,
                                   out_file_path=None,out_file_name=None):
    if(file_name.endswith('xslx')):
        data = pd.read_excel(path+'/'+file_name)
    elif(file_name.endswith('csv')):
        data = pd.read_csv(path+'/'+file_name)
    data_np = np.array(data)
    if(data_np.shape[1]!= 2):
        print('the data must have 2 cols')
        return
    score_col = data_np[:,0]
    label_col = data_np[:,1]
    score_min,score_max = score_col.min(),score_col.max()
    #if the min of the data > min or max of the data < max, replace the min and max
    if(score_min>min):
        min = int(score_min/split_range)*split_range
    elif(score_max<max):
        max = int(score_max/split_range+1)*split_range
    index_array = np.arange(min,max+split_range+1,split_range)
    out_over = []
    sum_over = []
    for index in index_array:
        overdue_list = []
        sum_list = []
        for i in range(score_col.shape[0]):
            if(score_col[i]>index and score_col[i] < (index+split_range)):
                if(label_col[i]==1):
                    overdue_list.append(label_col[i])
                sum_list.append(label_col[i])
        out_over.append(np.sum(overdue_list))
        sum_over.append(len(sum_list))
    overdue = np.array(out_over).reshape(-1,1)
    sum_all = np.array(sum_over).reshape(-1,1)
    ratio = (overdue/sum_all).reshape(-1,1)
    if(out_file_path is not None):
        out = np.concatenate((index_array.reshape(-1,1),sum_all,overdue,ratio),axis=1)
        out = pd.DataFrame(out)
        out.columns = ['range','sum_all','overdue','overdue_ratio']
        out.to_csv(out_file_path+'/'+out_file_name)
    else:
        return overdue,sum_all,ratio

path = 'F:\workingData\\201801/model_evaluation'
file_name = 'test_score.csv'
out_file_path = path
out_file_name = 'over_due_sum.csv'

overdue_produce_based_on_score(path,file_name=file_name,out_file_path=out_file_path,out_file_name=out_file_name,min=300)