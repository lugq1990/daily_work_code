# -*- coding:utf-8 -*-
"""This class is used for getting word2vec model matrix result, also can be choosen whether or not use tfidf algorithm.
    For training data, this is list type by using keywords extracted result.
"""
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import time

class word2vec_tfidf(object):
    """
        Parameters:
            word2vec_size: how many dimensions as result
            topk: how many keywords getted from sentence
            use_tfidf: whether or not to use tfidf for word2vec result(noted this is just based on corpus)
            word_min_count: minimum words selected
            word_iter: how many iteration for training word2vec model
            window: for how many windows to be choosen for training word2vec model
            word_sg: use distributed algorithm or use historical sampling.
            word_save_model: whether or not save the word2vec model to disk
            word_save_path: where to save the word2vec.
            workers: how many cores to be used for traing word2vec model
            silent: whether or not to print the process step.

    """
    def __init__(self, word2vec_size=100, topk=60, use_tfidf=True, word_min_count=1, word_iter=20, \
                 window=3, word_sg=1, word_save_model=False, word_save_path=None, workers=5,
                 silent=False):
        self.word2vec_size = word2vec_size
        self.use_tfidf  = use_tfidf
        self.min_count = word_min_count
        self.iter = word_iter
        self.window = window
        self.sg = word_sg
        self.save_model = word_save_model
        self.save_path = word_save_path
        self.silent = silent
        self.workers = workers
        self.topk = topk

    """
        This is used for getting IDF values in corpus.
        Parameters:
            data: which data to be processed.
        Returns:
            idf_dict: a dictory for each key: words, values: idf value in all corpus.
    """
    def _get_idf(self, data):
        # make all each getted keywords not use list as split, use blank as split for using tfidf
        tm = list()
        for i in range(len(data)):
            tm.append(' '.join(data[i]))

        idf_model = TfidfVectorizer(min_df=1)
        idf = idf_model.fit(tm).idf_
        idf_dict = dict(zip(idf_model.get_feature_names(), idf))

        return idf_dict

    """This is final tf-idf matrix result, multiply with word2vec result
        Parameters:
            word_dict: word2vec trained result dictory: key: words, value: vector getted from the Word2Vec model result.
            data: which data to be processed.
        Returns:
            result: tfidf value multiply with each word vector. This can be used for final result.
                Size: n*self.topk*self.word2vec_size
    """
    def _tfidf(self, word_dict, data):
        # Here I want to make tfidf result to be n*topk*word2vec_size, if word in idf-dictory, then all 3-D will be just same as tfidf
        result = np.empty((len(data), self.topk, self.word2vec_size), dtype=np.float32)

        # returned matrix is must be same like 'data', first compute each sentence 'tf' value, then multiply with getted words 'idf' value
        idf_dict = self._get_idf(data)

        idf_keys = set(idf_dict.keys())
        word_keys = set(word_dict.keys())

        # compute tfidf value, loop for each sentence and each keyword
        for i in range(len(data)):
            for j in range(len(data[i])):
                tf = data[i].count(data[i][j]) / len(data[i])
                # if keyword in this dict then use elementwise multiply with vector and the tfidf value vector(each row is same)
                if data[i][j] in idf_keys and data[i][j] in word_keys:
                    #                 tfidf[i][j] = tf* idf_dict[data[i][j]]
                    result[i][j][:] = tf * idf_dict[data[i][j]]
                    result[i][j][:] *= word_dict[data[i][j]]
                else:
                    # if the key not in idf directory keys, then just make tfidf value to be 1.
                    #                 tfidf[i][j] = 1.
                    result[i][j][:] = 0

        return result

    """this is a function for get the basic word2vec result matrix.
        Parameters:
            word_dict: word2vec trained result dictory: key: words, value: vector getted from the Word2Vec model result.
            data: which data to be processed.
        Returns:
            res: get Word2vec trained result with original datasets order vector.
                Size: n*self.topk*self.word2vec_size
    """
    def _get_basic_word2vec_matrix(self, word_dict, data):
        res = np.empty((len(data), self.topk, self.word2vec_size), dtype=np.float32)

        words_key = word_dict.keys()

        # for loop
        for r in range(len(data)):
            for r2 in range(len(data[r])):
                if data[r][r2] in words_key:
                    res[r][r2][:] = word_dict[data[r][r2]]
                else:
                    res[r][r2][:] = 0

        return res

    """This is main word2vec model with tfidf algorithm, returned is final result.
        Parameters:
            data: what data to be processed.
        Returns:
            result_returned: According to whether or not using TFIDF algorithm, return result.
            Size: n*self.topk*self.word2vec_size
    """
    def get_word2vec_tfidf(self, data):
        s_t = time.time()

        if not self.silent:
            print('Now is training for word2vec model')

        model = Word2Vec(data, min_count=self.min_count, iter=self.iter, size=self.word2vec_size, workers=self.workers,
                         window=self.window, sg=self.sg, max_vocab_size=None)
        if self.save_model:
            if self.save_path is not None:
                model.save(self.save_path + '/tmp_model/word2vec.bin', protocol=2)
            else:
                raise AttributeError('For saving model to disk, parameter save_path must be given!')

        wordsvec = model[model.wv.vocab]
        uni_words = list(model.wv.vocab)
        # construct a dict for word2vec result
        word_dict = dict()
        for m in range(len(uni_words)):
            word_dict[uni_words[m]] = wordsvec[m, :]

        if self.use_tfidf:
            if not self.silent:
                print('Now is using tfidf')
            # this result is n*60*100
            result_returned = self._tfidf(word_dict, data)
        else:
            if not self.silent:
                print('This is not using tfidf')
            # if do not use tfidf, then just make result to be 3-D, for just getted result
            result_returned = self._get_basic_word2vec_matrix(word_dict, data)

        e_t = time.time()
        if self.use_tfidf:
            print('Using TFIDF, get final result use {:.4f} seconds'.format((e_t - s_t)))
        else:
            print('Not using Tfidf, GET word2vec result use {:.4f} seconds'.format((e_t - s_t)))

        return result_returned


if __name__ == '__main__':
    import pandas as pd
    import jieba
    import jieba.analyse as ana

    topk = 20
    path = 'F:\working\\201808'
    df = pd.read_excel(path+'/pos_neg.xlsx')
    df.drop('company', axis=1, inplace=True)
    df.dropna(inplace=True)
    title = df.title

    def cutword(data):
        res = list()
        for i, d in enumerate(data):
            res.append(' '.join(jieba.cut(d)))
        return res

    def get_keyword(data):
        res = list()
        for i, d in enumerate(data):
            res.append(ana.extract_tags(d, topK=topk))
        return res

    cw = get_keyword(title)

    w2v = word2vec_tfidf(topk=topk, word2vec_size=20, silent=False, use_tfidf=True, word_iter=10)
    r = w2v.get_word2vec_tfidf(cw)
    print('Finished! result shape: ', r.shape)



