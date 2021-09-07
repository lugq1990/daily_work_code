# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

top = .9
wspace = .3

def plot_all_var_frequency(df, bins=15):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    df.hist(bins=bins, color='steelblue', edgecolor='black', linewidth=1.0,
            xlabelsize=8, ylabelsize=8, grid=False)
    plt.tight_layout()

    plt.show()


def plot_continuous_var_frequency_with_mu(df, col_name, bins=15, figsize=(8,6), title='Frequency ',
                                     title_fontsize='large', text_fontsize='medium'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')
    columns = df.columns
    if(col_name not in columns):
        raise ValueError('the column %s not in the DataFrame'%(col_name))
    data = df[col_name]
    if(data.dtype != np.float):
        raise TypeError('only the continuous column is supported.')

    fig = plt.figure(figsize=figsize)
    fig.suptitle(col_name+" "+title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Frequency')
    x_axis = np.max(data)
    y_axis = df.shape[0]/6

    ax.text(x_axis, y_axis, r'$\mu$='+str(round(data.mean(),2))+' $\sigma=$'+str(round(data.std(),2)),
            fontsize=text_fontsize, horizontalalignment='right')
    ax.hist(data, color='steelblue', bins=bins,
            edgecolor='black', linewidth=1.)

    plt.show()



def plot_descrite_vars_frequency(df, cols=None, figsize=(8,6), title='Frequency',
                             title_fontsize='large', text_fontsize='medium', cmap='nipy_spectral'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')
    if(cols is None):
        cols = df.columns.tolist()
    elif(isinstance(cols, np.str)):
        cols = [cols]
    elif(isinstance(cols, np.array)):
        cols = cols.tolist()
    else:
        raise TypeError('only the list-like columns is supported.')

    ax_list = list()
    data = df[cols]
    cols_new_list = list()
    for i in range(len(cols)):
        if(data[cols[i]].dtype != np.int64):
            data.drop(cols[i], axis=1, inplace=True)
        else:cols_new_list.append(cols[i])

    if(len(cols_new_list) == 0):
        raise ValueError('There is no descrite variable to plot.')
    elif(len(cols_new_list) >=5 ):
        over_times = len(cols_new_list) % 5
        figsize = (10+over_times*3,8)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    num = len(cols_new_list)
    for j in range(num):
        color = plt.cm.get_cmap(cmap)(float(j)/ num)
        ax_list.append(fig.add_subplot(1,num, j+1))
        ax_list[j].set_xlabel(cols_new_list[j], fontsize=text_fontsize)
        ax_list[j].set_ylabel('Frequency', fontsize=text_fontsize)
        data_values = data[cols_new_list[j]].value_counts()
        data_values = (list(data_values.index),list(data_values.values))
        ax_list[j].tick_params(axis='both', which='major', labelsize=8)
        ax_list[j].bar(data_values[0],data_values[1], color=color, edgecolor='black', linewidth=1)

    plt.show()

def plot_continuous_vars_distribution(df, cols=None, title='Given continuous variable distribution ',
                                     figsize=(10,8), bins=None, hist=True, rug=False,
                                     title_fontsize='large', text_fontsize='medium'):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    if(cols is None):
        cols = df.columns.tolist()
    elif(isinstance(cols,np.str)):
        cols = [cols]
    elif(isinstance(cols,np.array)):
        cols = cols.tolist()
    else:
        raise TypeError('only the list-like columns is supported.')

    ax_list = list()
    data = df[cols]
    cols_new_list = list()
    for i in range(len(cols)):
        if(data[cols[i]].dtype != np.float):
            data.drop(cols[i], axis=1, inplace=True)
        else:cols_new_list.append(cols[i])

    if(len(cols_new_list) ==0):
        raise ValueError('There is no continous column to plot.')
    elif(len(cols_new_list) >= 4 ):
        over_times = len(cols_new_list) % 4
        figsize = (10+over_times*3,8)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    num = len(cols_new_list)
    for i in range(num):
        ax_list.append(fig.add_subplot(1,num,i+1))
        ax_list[i].set_xlabel(cols_new_list[i], fontsize=text_fontsize)
        ax_list[i].set_ylabel('Frequency', fontsize=text_fontsize)
        plot_data = data[cols_new_list[i]]
        sns.distplot(plot_data, bins=bins, hist=hist, rug=rug)
    plt.show()


path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')
col = 'quality'
col2 = 'alcohol'
col3 = 'pH'
col4 = 'density'

#plot_descrite_vars_frequency(df,col)
plot_continuous_vars_distribution(df, col2)