# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time

"""
make the two file to be one file,use the first file as the based one,
the second file is used to match the first file, note the second file contains
the label cols at the last col, both of the them have to contain the same col for joint,
the first file contains all the needed features

params:
    combine_index_name:the col for joint
    start_index_name:start col for splitting of the first file
    end_index_name:end col for splitting of the first file
    need_to_save_to_file: whether to save the result to file,default to .csv file
    is_label_in_first_file:if the label col in the first file, default it is in the second file

    return:first cols is the combine_index_name, followed the features of the first file,
            last is the second file labels

"""

def combineTwoFile(path,file_1,file_2,
                   combine_index_name,
                   start_index_name,
                   end_index_name,
                   label_name='label',
                   need_to_save_to_file=False,
                   save_file_name=None,
                   is_label_in_first_file=False):
    time_start = time.time()
    if(file_2.endswith('xlsx') and file_1.endswith('xlsx')):
        df1 = pd.read_excel(path+"/"+file_1)
        df2 = pd.read_excel(path+"/"+file_2)
    if(file_1.endswith('csv') and file_2.endswith('csv')):
        df1 = pd.read_csv(path+"/"+file_1)
        df2 = pd.read_csv(path+"/"+file_2)
    print('load data finished!')
    #get both file combine cols data
    file_2_combine_df = df2[combine_index_name]
    file_1_combine_df = df1[combine_index_name]
    #use the python set for intersection
    same_list = list(set(file_1_combine_df).intersection(set(file_2_combine_df)))

    #print('the same row is ',same_list)
    cols_for_file_1 = df1.columns

    #make sure all the cols is string type
    cols_for_file_1 = cols_for_file_1.astype(np.str)
    df1.columns = cols_for_file_1


    #get the first file index of start and end index
    start_index = cols_for_file_1.get_loc(start_index_name)
    end_index = cols_for_file_1.get_loc(end_index_name)

    #get the start_index to end_index of label cols' names
    cols = cols_for_file_1[start_index:end_index+1]

    #out shape is the satisified rows length and needed columns + 2(combine_index_name and label)
    out = np.empty((len(same_list),end_index-start_index+2)).astype(np.str)

    #get the satisfied rows for df1 and df2
    sati_row_file_1 = df1[combine_index_name].map(lambda x:True if x in same_list else False)
    sati_row_file_2 = df2[combine_index_name].map(lambda x:True if x in same_list else False)

    #for the first col, it is the index col of the first file
    id_cols = df1.loc[sati_row_file_1,combine_index_name]
    #for the middle cols, it is the features needed,use pd.loc method, rows is the boolean type
    #start_index_name is the start index name
    out[:,0:(end_index-start_index)+1] = df1.loc[sati_row_file_1,start_index_name:end_index_name]
    #for the final col, it is the file_2 labels
    if(is_label_in_first_file):
        out[:,-1] = df1.loc[sati_row_file_1,label_name]
    else:
        out[:, -1] = df2.loc[sati_row_file_2, label_name]
    print('finished all procedure use %f seconds'%((time.time()-time_start)))
    #combine the index, features and label cols to be one for output
    result = np.concatenate((id_cols.values.reshape(-1,1),out),axis=1)
    #judge whethere or not to save the result to a file with cols' names
    if(need_to_save_to_file):
        res = pd.DataFrame(result)
        res_cols_list = []
        res_cols_list.append(combine_index_name)
        cols_list = list(cols)
        res_cols_list.extend(cols_list)
        res_cols_list.append(label_name)
        res.columns = res_cols_list
        res.to_csv(path+"/"+save_file_name,index=False)
    else:
        return result

path = 'F:\workingData\\201801\yidun_union'
file_1 = 'union_data.xlsx'
file_2 = 'yidun_data.xlsx'
combine_index_name = 'shddh'
#for the start_index and end_index is all in the first file
start_index_name = 'CSRL003'
end_index_name = 'CDTC015'
# start_index_name = '101'
# end_index_name = '154'

out = combineTwoFile(path,file_1,file_2,combine_index_name,start_index_name,end_index_name,
                     need_to_save_to_file=True,save_file_name='result.csv',is_label_in_first_file=False)