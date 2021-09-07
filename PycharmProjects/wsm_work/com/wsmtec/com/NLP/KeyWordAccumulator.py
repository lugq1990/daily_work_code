# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 08 10:28:46 2018
#
# @author: Administrator
# """
# import pandas as pd
# import numpy
# import time, os, pp
# import jieba
# import jieba.analyse
# import jieba.posseg as pseg
# import re
# import _pickle as pickle
# import gc
# from __future__ import print_function
#
# from itertools import chain
# from collections import Counter
# from math import log
# from multiprocessing import cpu_count
#
# from warnings import filterwarnings
#
# filterwarnings('ignore')
# BASE_DIR = os.path.dirname(__file__)
# DICT_DIR = os.path.join(BASE_DIR, "dict_new.txt")
#
# '''
# Ag	形语素	形容词性语素。形容词代码为 a，语素代码ｇ前面置以A。
# a	形容词	取英语形容词 adjective的第1个字母。
# ad	副形词	直接作状语的形容词。形容词代码 a和副词代码d并在一起。
# an	名形词	具有名词功能的形容词。形容词代码 a和名词代码n并在一起。
# b	区别词	取汉字“别”的声母。
# c	连词	取英语连词 conjunction的第1个字母。
# dg	副语素	副词性语素。副词代码为 d，语素代码ｇ前面置以D。
# d	副词	取 adverb的第2个字母，因其第1个字母已用于形容词。
# e	叹词	取英语叹词 exclamation的第1个字母。
# f	方位词	取汉字“方”
# g	语素	绝大多数语素都能作为合成词的“词根”，取汉字“根”的声母。
# h	前接成分	取英语 head的第1个字母。
# i	成语	取英语成语 idiom的第1个字母。
# j	简称略语	取汉字“简”的声母。
# k	后接成分
# l	习用语	习用语尚未成为成语，有点“临时性”，取“临”的声母。
# m	数词	取英语 numeral的第3个字母，n，u已有他用。
# Ng	名语素	名词性语素。名词代码为 n，语素代码ｇ前面置以N。
# n	名词	取英语名词 noun的第1个字母。
# nr	人名	名词代码 n和“人(ren)”的声母并在一起。
# ns	地名	名词代码 n和处所词代码s并在一起。
# nt	机构团体	“团”的声母为 t，名词代码n和t并在一起。
# nz	其他专名	“专”的声母的第 1个字母为z，名词代码n和z并在一起。
# o	拟声词	取英语拟声词 onomatopoeia的第1个字母。
# p	介词	取英语介词 prepositional的第1个字母。
# q	量词	取英语 quantity的第1个字母。
# r	代词	取英语代词 pronoun的第2个字母,因p已用于介词。
# s	处所词	取英语 space的第1个字母。
# tg	时语素	时间词性语素。时间词代码为 t,在语素的代码g前面置以T。
# t	时间词	取英语 time的第1个字母。
# u	助词	取英语助词 auxiliary
# vg	动语素	动词性语素。动词代码为 v。在语素的代码g前面置以V。
# v	动词	取英语动词 verb的第一个字母。
# vd	副动词	直接作状语的动词。动词和副词的代码并在一起。
# vn	名动词	指具有名词功能的动词。动词和名词的代码并在一起。
# w	标点符号
# x	非语素字	非语素字只是一个符号，字母 x通常用于代表未知数、符号。
# y	语气词	取汉字“语”的声母。
# z	状态词	取汉字“状”的声母的前一个字母。
# un	未知词	不可识别词及用户自定义词组。取英文Unkonwn首两个字母。(非北大标准，CSW分词中定义)
# '''
#
# class cured_jieba():
#
#     def __init__(self, stopword_dict_path=None, dictionary=None, restricted_lib=None):
#         self.part = None
#         self._stopwords = None
#         self._col_name = None
#         self._dict_dir = dictionary
#         self._restricted_lib = restricted_lib
#         self._mode = ""
#         if stopword_dict_path:
#             self._set_stopwords(stopword_dict_path=stopword_dict_path)
#
#     def _combine(self, part):
#         self.part = pd.concat([self.part, part])
#
#     def _set_stopwords(self, stopword_dict_path=None):
#         fl = open(stopword_dict_path, 'r')
#         self._stopwords = [w.strip().decode("utf8") for w in fl]
#         fl.close()
#
#     def _clean_words(self, segwords):
#         if self._stopwords:
#             ret = [wd for wd in segwords if not wd.strip() in self._stopwords and len(wd.strip()) > 1]
#             segwords = None
#             segwords = ret
#         if self._restricted_lib:
#             ret = [wd for wd in segwords if wd.strip() in self._restricted_lib and len(wd.strip()) > 1]
#             segwords = None
#             segwords = ret
#         ret = filter(lambda z: True if not re.findall("[0-9a-zA-Z]+", z) and z else False, segwords)
#         return ret
#
#     def _tmp_func(self, df):
#         if self._dict_dir:
#             print "%s INFO: setting dictionary at subprocess" %time.ctime()
#             jieba.load_userdict(self._dict_dir)
#         if self._mode == "cut":
#             df['segword'] = df[self._col_name].apply(self._cut_words)
#         elif self._mode == "tags":
#             df['segword'] = df[self._col_name].apply(self._extract_tags)
#         elif self._mode == "search":
#             df['segword'] = df[self._col_name].apply(self._cut_for_search)
#         elif self._mode == "flag":
#             df['segword'] = df[self._col_name].apply(self._extract_tags_by_flag)
#         return df
#
#     def _parallel(self, data):
#         assert isinstance(data, pd.io.parsers.TextFileReader), "TypeError: a TextFileReader required"
#         assert self._col_name != None, 'target_col can not be empty!'
#         ppservers = ()
#         ncpus = cpu_count() - 1
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#         for dat in data:
#             d = dat[[self._col_name]]
#             job_server.submit(self._tmp_func, (d, ), (), ("jieba", "re", "pandas", "numpy", "time"), callback=self._combine)
#         #wait for jobs in all groups to finish
#         job_server.wait()
#         job_server.print_stats()
#
#     def _extract_tags(self, orig_word):
#         tags = jieba.analyse.extract_tags(orig_word.strip(), topK=self._topK, allowPOS=self._allowPOS)
#         return list(tags)
#
#     def extract_tags(self, data, target_col, topK=None, allowPOS=None, stop_words=None):
#         #allowPOS=['ns', 'n', 'vn', 'v','nr']
#         assert allowPOS != None, 'allowPOS can not be empty!'
#         self._col_name = target_col
#         self._topK = topK
#         self._allowPOS = allowPOS
#         if stop_words:
#             jieba.analyse.set_stop_words(stop_words)
#         self._mode = "tags"
#         self._parallel(data)
#         return self.part.sort_index()
#
#     def _cut_for_search(self, orig_word):
#         segwords = jieba.cut_for_search(orig_word.strip())  # 搜索引擎模式
#         ret = self._clean_words(segwords)
#         return ret
#
#     def cut_for_search(self, data, target_col):
#         self._col_name = target_col
#         self._mode = "search"
#         self._parallel(data)
#         return self.part.sort_index()
#
#     def _extract_tags_by_flag(self, orig_word):
#         segwords = []
#         words = pseg.cut(orig_word)
#         for word, flag in words:
#             if flag in self._flags:
#                 segwords.append(word)
#         ret = self._clean_words(segwords)
#         return ret
#
#     def extract_tags_by_flag(self, data, target_col, flags=None):
#         #flags=['ns', 'n', 'vn', 'nr']
#         assert flags != None, 'flags can not be empty!'
#         self._col_name = target_col
#         self._mode = "flag"
#         self._flags = flags
#         self._parallel(data)
#         return self.part.sort_index()
#
#     def _cut_words(self, orig_word):
#         segwords = jieba.cut(orig_word.strip(), cut_all=self._cut_all)
#         ret = self._clean_words(segwords)
#         return ret
#
#     def cut(self, data, target_col, cut_all=False):
#         self._col_name = target_col
#         self._mode = "cut"
#         self._cut_all = cut_all
#         self._parallel(data)
#         return self.part.sort_index()
#
#
# class WordDog():
#
#     def __init__(self, max_iter=5, chunksize=10, thredsholds=[50., 0.5], save_proc=False):
#         self.searched_words = []
#         self._corpus_size = 0
#         self._thredsholds = thredsholds
#         self._max_iter = max_iter
#         self._chunksize = chunksize
#         self.part = None
#         self._save_proc = save_proc
#
#     def _filter_word_list(self, segwords):
#         ret = [wd for wd in segwords if wd.strip() in self.searched_words]
#         return ret
#
#     def _cal_tfidf(self, uni_word_count):
#         def tfidf(word_dict):
#             lib = {}
#             for k, e in word_dict.iteritems():
#                 lib[k] = e / float(len(word_dict.values())) * log(self._corpus_size / (uni_word_count[e] + 1))
#             return Counter(lib)
#         return tfidf
#
#     def _collect_keywords(self, segwords):
#         ret = Counter(self._filter_word_list(segwords))
#         return ret.items()
#
#     def search(self, corpus=None, target_col=None):
#         assert target_col != None, 'target_col can not be empty!'
#         assert isinstance(corpus, pd.DataFrame), 'corpus must be a DataFrame!'
#         self._corpus_size = corpus.shape[0]
#         _corpus = corpus.copy()
#         a_tfidf = self._thredsholds[1]
#         #self.col_name = target_col
#         words = _corpus[target_col].tolist()
#         for i in xrange(self._max_iter):
#             #统计每行中各单词出现的次数，并保存成字典
#             print u"%s INFO: Iteration %d 统计每行中各单词出现的次数" %(time.ctime(), i+1)
#             word_cnt_perRow = map(lambda w: Counter(w), words)
#             #将words列表中的各个列表拆散成一个大列表
#             all_words = list(chain.from_iterable(words))
#             #统计每个词的出现次数，保存成字典。相当于统计各个词总的出现次数
#             print u"%s INFO: Iteration %d 统计每个词的出现次数" %(time.ctime(), i+1)
#             word_count = Counter(all_words)
#             #每行单词表进行去重
#             print u"%s INFO: Iteration %d 每行单词表进行去重" %(time.ctime(), i+1)
#             uni_words_perRow = map(lambda w: list(set(w)), words)
#             #将去重后的uni_words_perRow列表中的各个列表拆散成一个大列表
#             uni_word_lib = list(chain.from_iterable(uni_words_perRow))
#             #统计每个词的出现次数，保存成字典。相当于统计各个词在不同文章中出现的次数
#             print u"%s INFO: Iteration %d 统计每个词的出现次数" %(time.ctime(), i+1)
#             uni_word_count = Counter(uni_word_lib)
#             #计算词的TF-IDF得分
#             print u"%s INFO: Iteration %d 计算词的TF-IDF得分" %(time.ctime(), i+1)
#             cal_tfidf = self._cal_tfidf(uni_word_count)
#             self.word_tfidf = map(cal_tfidf, word_cnt_perRow)
#
#             word_dict = {}
#             word_dict["word"] = []
#             word_dict["count"] = []
#             word_dict["docs"] = []
#             for w in word_count.keys():
#                 word_dict["word"].append(w)
#                 word_dict["count"].append(word_count[w])
#                 word_dict["docs"].append(uni_word_count[w])
#             word_tbl = pd.DataFrame(word_dict, columns=["word", "count", "docs"])
#             self.total = numpy.sum((numpy.array(word_dict["docs"]) * 1.))
#
#             self._cal_weight(word_tbl)
#             del word_tbl
#             gc.collect()
#             word_tbl = self.part.copy().sort_index()
#             del self.part
#             gc.collect()
#             self.part = None
#
#             tmp = word_tbl[(word_tbl["weight"]>self._thredsholds[0]) & (word_tbl["avg_tfidf"]>a_tfidf)]
#             del word_tbl
#             gc.collect()
#             a_tfidf += self._thredsholds[1]
#             self.searched_words = tmp["word"].tolist()
#             print "%s INFO: filtered %d keywords" %(time.ctime(), len(self.searched_words))
#             if self._save_proc:
#                 print "%s INFO: Saving temporary table" %time.ctime()
#                 f = file("searched_words%d.pkl" %(i + 1), "wd")
#                 pickle.dump(self.searched_words, f)
#                 f.close()
#             tmp = map(self._filter_word_list, words)
#             words = tmp
#             del tmp
#             gc.collect()
#         _corpus[target_col] = _corpus[target_col].apply(self._collect_keywords)
#         return _corpus
#
#     def _cal_weight(self, data):
#         ppservers = ()
#         ncpus = cpu_count() - 1
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#         for dat in self._iterating(data):
#             job_server.submit(self._tmp_func, (dat,), (), ("pandas", "numpy", "time"), callback=self._combine)
#         job_server.wait()
#         job_server.print_stats()
#
#     def _tmp_func(self, df):
#         print "%s INFO: handling %d words" %(time.ctime(), len(df))
#         df["tfidf_list"] = df["word"].apply(lambda w: list(set(map(lambda z: z.get(w), self.word_tfidf)) - set([None])))
#         #计算各个词的平均TF-IDF得分
#         df["avg_tfidf"] = df["tfidf_list"].apply(lambda z: sum(z) / len(z))
#         #计算各个词在整个语料数据中权重
#         df["weight"] = -1 * df["count"] * numpy.log((df["docs"] * 1.) / self.total)
#         return df
#
#     def _iterating(self, df):
#         max_column = df.shape[0]
#         number = max_column / self._chunksize if max_column % self._chunksize == 0 else max_column / self._chunksize + 1
#         start = 0
#         for i in xrange(self._chunksize):
#             end = start + number
#             d = df.iloc[start:end,:]
#             start = end
#             yield d
#
#     def _combine(self, part):
#         self.part = pd.concat([self.part, part])
#
#
# if __name__ == "__main__":
#     start = time.time()
#     print "Starting at %s" %time.ctime()
#     STOPWORDS_DIR = os.path.join(BASE_DIR, "stop_words_cmp.txt")
#
#     WORD_DIR = os.path.join(BASE_DIR, "content.csv")
#     #WORD_DIR = os.path.join(BASE_DIR, "segword.csv")
#     WORDS_DIR = os.path.join(BASE_DIR, "words_level3.txt")
#
#     partial_data = pd.read_csv(WORD_DIR, encoding="gb18030", chunksize=2000)
#
#
#     cutter = cured_jieba(stopword_dict_path=STOPWORDS_DIR, DICT_DIR=DICT_DIR)
#     segword = cutter.cut(partial_data, "content")
#     #f = file("segword.pkl", "w")
#     #pickle.dump(segword, f)
#     #f.close()
#     #segword.to_csv("segword.csv", encoding="gb18030", index=False)
#
#
#     #f = file("segword.pkl", "r")
#     #segword = pickle.load(f)
#     #f.close()
#
#     #segword = pd.read_csv(WORD_DIR, encoding="gb18030")
#     wdg = WordDog(chunksize=3)#thredsholds=[25., 0.1])
#     segword["keywords"] = wdg.search(segword[["segword"]], "segword")
#     f = file("segword.pkl", "w")
#     pickle.dump(segword, f)
#     f.close()
#     print "Stopping at %s" %time.ctime()
#     print time.time() - start
#
#
#
#
#
#
