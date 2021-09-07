# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')
top = .9
wspace = .3

def plot_heat_map(df, figsize=(10,8), title='Correlation Heatmap',
                  title_fontsize='large'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    fig = plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(round(corr,2), annot=True, cmap='coolwarm', fmt='.2f',
                     linewidths=.05)
    fig.subplots_adjust(top=top, wspace=wspace)
    fig.suptitle(title, fontsize=title_fontsize)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def plot_pairwise(df, cols, title='Attribute Pairwise',
                  title_font_size='large', text_fontsize='medium'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    columns = df.columns
    if(len(cols) > 8):
        raise UserWarning('For now just use the col length< 8 will be supported')
    else:
        for i in range(len(cols)):
            if(cols[i] not in columns):
                raise ValueError('the column %s not in the DataFrame'%(cols[i]))
        data = df[cols]

        pp = sns.pairplot(data, size=1.8, aspect=1.8, plot_kws=dict(edgecolor='k',linewidth=.5),
                     diag_kind='kde', diag_kws=dict(shade=True))
        fig = pp.fig
        fig.subplots_adjust(top=top, wspace=wspace)
        fig.suptitle(title, fontsize=title_font_size)

        plt.show()

def plot_two_continues_scatter(df, col_1, col_2, alpha=.4, title='Pair',
                               title_fontsize='large', text_fontsize='medium'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    data_1 = df[col_1]
    data_2 = df[col_2]
    if(data_1.dtype != np.float and data_2.dtype != np.float):
        raise TypeError('only the continuous column is supported.')

    plt.scatter(data_1, data_2, alpha=alpha, edgecolors='w')
    plt.xlabel(col_1, fontsize=text_fontsize)
    plt.ylabel(col_2, fontsize=text_fontsize)
    plt.title(col_1+" and "+col_2+" "+title, fontsize=title_fontsize)
    plt.show()


def plot_joint(df, col_1, col_2):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    sns.jointplot(x=col_1, y=col_2, data=df,
                  kind='reg', space=0, size=5, ratio=4)
    plt.show()


def plot_descrite_frequency_based_label(df, col, label,
                                        title='Same Variable Frequency Based on Label',figsize=(10,8),
                                            title_fontsize='large', text_fontsize='medium',
                                            cmap='nipy_spectral'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    data = df[col]
    if(data.dtype != np.int64):
        raise TypeError('only the descrite column supported ')

    labels = df[label]
    unique_label = np.unique(labels)
    df_new = df[[col,label]]

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    num = unique_label.shape[0]
    ax_list = list()
    for i in range(num):
        #fig.add_subplot(1,num,i+1)
        ax_list.append(fig.add_subplot(1,num,i+1))
        color = plt.cm.get_cmap(cmap)(float(i) / num)
        # fig,ax = plt.subplots(1,num,i)
        ax_list[i].set_title(unique_label[i], fontsize=title_fontsize)
        ax_list[i].set_xlabel(col, fontsize=text_fontsize)
        ax_list[i].set_ylabel('Frequency', fontsize=text_fontsize)
        #get the satisified data depend on different labels
        data = df_new[df_new.label==unique_label[i]][col].value_counts()
        data_s = (list(data.index),list(data.values))
        ax_list[i].tick_params(axis='both', which='major', labelsize=8)
        ax_list[i].bar(data_s[0], data_s[1], color=color, edgecolor='black', linewidth=1.)

    plt.show()

def plot_continus_density_based_label(df, col, label, title='Same Variable Density Based on Label ',
                                      figsize=(10,8),title_fontsize='large', text_fontsize='medium',
                                      cmap='nipy_spectral'):
    if(not isinstance(df,pd.core.frame.DataFrame)):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    data = df[col]
    if(data.dtype != np.float):
        raise TypeError('only the continus column supported ')

    labels = df[label]
    unique_label = np.unique(labels)
    df_new = df[[col, label]]

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)

    num = unique_label.shape[0]
    ax_list = list()
    for i in range(num):
        ax_list.append(fig.add_subplot(1, num, i+1 ))
        color = plt.cm.get_cmap(cmap)(float(i)/num)
        ax_list[i].set_title(unique_label[i], fontsize=title_fontsize)
        ax_list[i].set_xlabel(col, fontsize=text_fontsize)
        ax_list[i].set_ylabel('Density', fontsize=text_fontsize)
        data = df_new[df_new.label == unique_label[i]][col].value_counts()
        #data_s = (list(data.index),list(data.values))
        ax_list[i].tick_params(axis='both', which='major', labelsize=8)
        sns.kdeplot(data, ax=ax_list[i], shade=True, color=color)

    plt.show()

def plot_continues_frequency_based_label_one_fig(df, col, label, figsize=(10,8), bins=15,
                                              title='Continues Variable Frequency Based On Label',
                                              title_fontsize='large', text_fontsize='medium',
                                              cmap='nipy_spectral'):
    if not isinstance(df,pd.core.frame.DataFrame):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')

    data = df[col]
    if(data.dtype != np.float):
        raise TypeError('only the continus column supported ')

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)
    fig.subplots_adjust(top=top, wspace=wspace)
    ax.set_xlabel(col, fontsize=text_fontsize)
    ax.set_ylabel('Frequency', fontsize=text_fontsize)

    labels = df[label]
    unique_label = np.unique(labels)

    palette = dict()
    for i in range(len(unique_label)):
        color = plt.cm.get_cmap(cmap)(float(i) / len(unique_label))
        palette[unique_label[i]] = color

    facegrid =sns.FacetGrid(df, hue='label', palette=palette)
    facegrid.map(sns.distplot, col, kde=True, bins=bins, ax=ax)
    ax.legend(title='label')

    plt.show()

def plot_kernel_density_estimation(df, col1, col2, title='Kernel Density Estimation',
                                   figsize=(10,8), title_fontsize='large',
                                   text_fontsize='medium'):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')
    if(df[col1].dtype != np.float or df[col2].dtype != np.float):
        raise TypeError('the columns must be continues.')

    sns.jointplot(x=col1, y=col2, data=df, kind='kde')
    plt.show()

def plot_more_continuously_density_for_binary(df, col1, col2, title='More Density plot',
                                              figsize=(10,8), n_levels=60, shade=True,
                                              title_fontsize='large'):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('only DataFrame type is supported. please use the DataFrame type')
    if(df[col1].dtype != np.float or df[col2].dtype != np.float):
        raise TypeError('the columns must be continues.')

    f, ax = plt.subplots(figsize=figsize)
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(df[col1], df[col2], cmap=cmap, n_levels=n_levels, shade=shade)
    ax.set_title(title, fontsize=title_fontsize)
    plt.show()


a = np.ones((200,1))
b = np.zeros((200,1))
c = np.ones((df.shape[0]-400,1))*2
tmp = np.concatenate((a,b,c),axis=0)
df2 = pd.concat([df,pd.DataFrame(tmp,columns=['label'])],axis=1)

cols = ['pH','density','quality']
columns = df.columns
# plot_heat_map(df)
# plot_pairwise(df,cols)
#plot_two_continues_scatter(df,'pH','density')
#plot_joint(df,columns[0],columns[3])
#plot_descrite_frequency_based_label(df2,cols[-2],'label')
#plot_continus_density_based_label(df2,cols[0],'label')
# plot_continues_frequency_based_label_one_fig(df2, cols[2], 'label')
# plot_kernel_density_estimation(df,cols[0],cols[1])
# plot_more_continuously_density_for_binary(df,cols[0],cols[1])
plot_heat_map(df)