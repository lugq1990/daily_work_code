# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
import jieba
import jieba.analyse as ana
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import *
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
mpl.rcParams['font.sans-serif'] = ['SimHei']

connection = pymysql.connect(user='zhanghui',password='zhanghui',database='model_data',host='10.1.36.18',charset='utf8')
query = 'select item_name,if(label is null,0,1) as label from model_data.tb_item_label_rfm2 where sorttime <= \'2018-03-01\''
re = pd.read_sql(query,con=connection)
df = re[['item_name','label']]
df_np = np.array(df.dropna()).astype(np.str)

#first to make the 2-classes classfication
df_np = df_np[:1000,:]
#get the key words
data = df_np[:,0]
label = df_np[:,1]
def get_keywords(data=data,topk=4):
    key_words = list()
    for i,d in enumerate(data):
        key_words.append(ana.extract_tags(d,topK=topk,allowPOS=('n','nr')))
        if(i%20000 == 0):
            print('now is the %d samples'%(i))
    return key_words

key_words = get_keywords()

#make the key-words using word2vec
def get_word2vec(sentences=key_words,min_count=1,iter=5000,size=20):
    model = Word2Vec(sentences,min_count=min_count,iter=iter,size=size)
    wordsvec = model[model.wv.vocab]
    uni_words = list(model.wv.vocab)
    return wordsvec,uni_words
#get the word2vec result and the unique words list
wordsvec,uni_words = get_word2vec()

#make the key-words and word2vec result directory for bellow using the key-value
res_dic = dict()
for j in range(len(uni_words)):
    res_dic.update({uni_words[j]:wordsvec[j,:]})

#make the key_words sentences vecter by meaning the sum of the all vectors number
result = np.zeros((len(key_words),wordsvec.shape[1]))
#loop the all visuable vectors
for i in range(result.shape[0]):
    for it in key_words[i]:
        result[i,:] += res_dic[it]
    #average the result for each row as the each sentence vector
    result[i,:] = result[i,:]/len(key_words[i])

#train_data = np.concatenate((result,label.reshape(-1,1)),axis=1)




#define the method to plot the key-words
def plot_word2vec(sentences=data,topk=5,min_count=2,size=50,iter=10000,needed_plot_ratio=.01):
    #random get the given data size words
    total_num = sentences.shape[0]
    rand_num = np.random.choice(a=(True,False),size=(total_num),p=(needed_plot_ratio,(1-needed_plot_ratio)))
    sentences = sentences[rand_num]
    #first to extract the key-words
    key_words = list()
    for i,d in enumerate(sentences):
        key_words.append(ana.textrank(d,topK=topk,allowPOS=('n')))

    model = Word2Vec(sentences=key_words,min_count=min_count,
                     size=size,iter=iter)
    words = list(model.wv.vocab)
    res = model[model.wv.vocab]
    res_pca = PCA(n_components=2).fit_transform(res)
    plt.scatter(res_pca[:,0],res_pca[:,1])
    #annotate the words
    for i,word in enumerate(words):
        plt.annotate(word,xy=(res_pca[i,0],res_pca[i,1]))
    plt.show()



