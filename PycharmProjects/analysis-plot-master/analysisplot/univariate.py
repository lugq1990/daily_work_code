# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

top = .9
wspace = .3

def plot_data_bar(df, bins=15):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    df.hist(bins=bins, color='steelblue', edgecolor='black', linewidth=1.0,
            xlabelsize=8, ylabelsize=8, grid=False)
    plt.tight_layout()

    plt.show()

def plot_continuous_var_distribution_with_mu(df, col_name, bins=15, figsize=(6,4), title='Distribution',
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
    ax.set_ylabel('Density')

    ns,_,_ = ax.hist(data, color='steelblue', bins=bins,
            edgecolor='black', linewidth=1.)
    axis = [data.max(),ns.max()]
    ax.text(axis[0], axis[1], r'$\mu$='+str(round(data.mean(),2))+' $\sigma=$'+str(round(data.std(),2)),
            horizontalalignment='right',verticalalignment='top',
            fontsize=text_fontsize)

    plt.show()

# def plot_single_continuous_distribution(df, col_name, figsize=(6,4), title='Distribution',
#                                    title_fontsize='large', text_fontsize='medium'):
#     if(not isinstance(df,pd.core.frame.DataFrame)):
#         raise TypeError('only DataFrame type is supported. please use the DataFrame type')
#     columns = df.columns
#     if(col_name not in columns):
#         raise ValueError('the column %s not in the DataFrame'%(col_name))
#     data = df[col_name]
#     if(data.dtype != np.float):
#         raise TypeError('only the continuous column is supported.')
#
#     fig = plt.figure(figsize=figsize)
#     fig.suptitle(col_name+" "+title, fontsize=title_fontsize)
#     fig.subplots_adjust(top=top, wspace=wspace)
#
#     ax = fig.add_subplot(1,1,1)
#     ax.set_xlabel(col_name)
#     ax.set_ylabel('Density')
#     sns.kdeplot(data, ax=ax, shade=True, color='steelblue')
#
#     plt.show()


def plot_descrite_vars_frequency(df, cols=None, figsize=(6,4), title='Frequency',
                             title_fontsize='large', text_fontsize='medium',
                             cmap='nipy_spectral'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    if(cols is None):
        cols = df.columns
    elif(isinstance(cols,np.str)):
        cols = [cols]
    elif(not isinstance(cols,list)):
        raise TypeError("Multi vars given type must be list type.")

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    cols = np.array(cols)
    num = cols.shape[0]
    ax_list = list()
    data_cols = df[cols]
    for j in range(num):
        if data_cols.iloc[:,j].dtype != np.int64:
            raise TypeError('Only the continuouse cols is supported, the cols {0:s} is discrete'.format(cols[j]))

    for i in range(num):
        color = plt.cm.get_cmap(cmap)(float(i) / num)
        ax_list.append(fig.add_subplot(1,num,i+1))
        ax_list[i].set_xlabel(cols[i], fontsize=text_fontsize)
        ax_list[i].set_ylabel('Frequency', fontsize=text_fontsize)
        ax_list[i].tick_params(axis='both', which='major', labelsize=8)
        data = df[cols[i]].value_counts()
        data_s = (list(data.index), list(data.values))
        ax_list[i].bar(data_s[0], data_s[1], color=color, edgecolor='black', linewidth=1)

    plt.show()



def plot_continuous_vars_distribution(df, cols=None, title='Distribution',
                                     figsize=(10,8),
                                     title_fontsize='large', text_fontsize='medium'):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('Only DataFrame type is supported. please use the DataFrame type')

    if(cols is None):
        cols = df.columns
    elif(isinstance(cols,np.str)):
        cols = [cols]
    elif(not isinstance(cols,list)):
        raise TypeError("Multi vars given type must be list type.")

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    cols = np.array(cols)
    num = cols.shape[0]
    ax_list = list()
    data_cols = df[cols]
    for j in range(num):
        if data_cols.iloc[:,j].dtype != np.float:
            raise TypeError('Only the continuouse cols is supported, the cols {0:s} is discrete'.format(cols[j]))

    for i in range(num):
        ax_list.append(fig.add_subplot(1,num,i+1))
        ax_list[i].set_xlabel(cols[i], fontsize=text_fontsize)
        ax_list[i].set_ylabel('Density', fontsize=text_fontsize)
        data = df[cols[i]]
        sns.distplot(data)
    plt.show()

path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')
col = 'quality'
col2 = 'pH'
col3 = 'alcohol'
col4 = 'density'
#plot_single_descrite_frequency(df,col)
# plot_given_variable_distribution(df,cols=['pH',col3,col4])
# plot_data_bar(df)
# plot_single_continuous_frequency_with_mu(df,col2,bins=20)
# plot_single_continuous_distribution(df,col4)
#plot_data_bar(df)
# plot_continuous_distribution(df,cols=[col4,col3])
# plot_continuous_var_distribution_with_mu(df,col4)
plot_descrite_vars_frequency(df,col)