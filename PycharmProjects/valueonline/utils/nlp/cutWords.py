# -*- coding:utf-8 -*-

import jieba
import jieba.analyse as ana
from jieba.analyse import textrank
import jieba.posseg as pseg

class cutWords(object):
    def __init__(self):
        pass

    # this is just for using exact solution to cut words.
    def cut_words_basic(self, data):
        res = list()
        step_nums = int(len(data)/10)
        for i, d in enumerate(data):
            res.append(' '.join(jieba.cut(d)))
            if i % step_nums == 0:
                print('Now is finished {:02d}% '.format(int(i/step_nums)*10))
        return res

    # cut word according to words attribution,
    def cut_words_attr(self, data, allowPos=['n','nr','nt', 'nz','ns'], HMM=True):
        res = list()
        step_nums = int(len(data)/10)
        for i, d in enumerate(data):
            cutted = pseg.cut(d, HMM=HMM)
            sati_list = list()
            for word, flag in cutted:
                if flag not in allowPos:
                    continue
                sati_list.append(word)
            if i % step_nums == 0:
                print('Now is Finished {:02d}% '.format(int(i/step_nums)*10))
            res.append(sati_list)
        return res

    # get keywords using TFIDF or textRank algorithm
    def get_keywords(self, data, topk=60, use_tfidf=True, allowPos=['n','nr','nt', 'nz','ns']):
        res = list()
        step_nums = int(len(data)/10)
        for i, d in enumerate(data):
            if use_tfidf:
                res.append(ana.extract_tags(d, topK=topk, allowPOS=allowPos))
            else:
                res.append(textrank(d, topK=topk, allowPOS=allowPos))
            if i % step_nums == 0:
                print('Now is Finished {:02d}%'.format(int(i/step_nums)*10))
        return res




if __name__ == '__main__':
    import pandas as pd
    path = 'F:\working\\201808'
    df = pd.read_excel(path+'/pos_neg.xlsx')
    df.drop('company', inplace=True, axis=1)
    df.dropna(inplace=True)

    title = df.title
    res = cutWords().get_keywords(title, topk=10)
    print(len(res))
    print(res[:4])
