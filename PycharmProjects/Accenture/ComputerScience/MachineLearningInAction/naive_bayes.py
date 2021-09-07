# -*- coding:utf-8 -*-
"""
This is based on naive bayes algorithms

@author: Guangqiang.lu
"""
import numpy as np


def load_dataset():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec


def create_vocab(dataset):
    """this is to get whole words"""
    vocab_set = set()
    for words in dataset:
        vocab_set = vocab_set.union(set(words))
    return list(vocab_set)


def create_word2vec(input_data, vocab_set):
    """this is to create a vector based on input data and vocab set"""
    return_vec = [0] * len(vocab_set)
    for word in input_data:
        if word in vocab_set:
            return_vec[vocab_set.index(word)] = 1
        else:
            pass
    return return_vec


def train_nb(train_matrix, train_label):
    n_docs = len(train_matrix)
    n_words = len(train_matrix[0])
    # compute class 1 prob
    p_class = sum(train_label) / n_docs
    # init words number
    p0_nums = np.zeros(n_words)
    p1_nums = np.zeros(n_words)
    # how much words for each document
    p0_words = 0.
    p1_words = 0.
    for i in range(n_docs):
        if train_label[i] == 1:
            p1_nums += train_matrix[i]
            p1_words += sum(train_matrix[i])
        else:
            p0_nums += train_matrix[i]
            p0_words += sum(train_matrix[i])
    p1_vector = p1_nums / p1_words
    p0_vector = p0_nums / p0_words
    return p0_vector, p1_vector, p_class


if __name__ == "__main__":
    dataset, labels = load_dataset()
    vocab_set = create_vocab(dataset)
    # print(create_word2vec(dataset[0], vocab_set))
    train_matrix = []
    for doc in dataset:
        train_matrix.append(create_word2vec(doc, vocab_set))

    p0, p1, pclass = train_nb(train_matrix, labels)
    print("class label:", p0)
    print("pclass:", pclass)
