# -*- coding:utf-8 -*-
"""This is a main class for implement skip-gram model training"""
import numpy as np
import random

class SkipGram(object):
    def __init__(self, n_dim=5, slide_window=3, n_negative=2, n_iter=10, seperator=' '):
        self.n_dim = n_dim
        self.slide_window = slide_window
        self.n_negative = n_negative
        self.n_iter = n_iter
        self.seperator = seperator

    def _logisic_function(self, x):
        return 1. / (1. + np.exp(-x))

    # Here I write the make data function to avoid too many in the train function,
    # return result is the list with both right words part also with negative sampling data
    def _get_train_array(self, data_list, uni_words):
        # according to slide window and negative sampling number to build the training word data
        # loop for every sentence
        res_list = []

        for data in data_list:
            data = data.split(self.seperator)
            # the index point
            index = int(self.slide_window / 2)

            if len(data) < self.slide_window or len(data_list) == 1:
                for i in reversed(range(1, len(data))):
                    res_list.append([data[i], data[i-1], 1])
                    # if the data is lower than sliding window, then here doesn't add negative data
            else:
                for j in range(self.slide_window - index - 1, len(data) - index):
                    # first add the right words, with the center words +/- int(slide_window/2) as neighbor words
                    for k in range(1, index + 1):
                        res_list.append([data[j], data[j - k], 1])
                        res_list.append([data[j], data[j + k], 1])
                    # here should add some negative data, here should add some words not in the sentence
                    not_apear_words = list(set(uni_words) - set(data))
                    # add how many negative according to the negative numbers
                    for _ in range(self.n_negative):
                        res_list.append([data[j], random.choice(not_apear_words), 0])

        return np.array(res_list)


    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif not isinstance(data, list):
            raise TypeError('Here just with numpy array or list')

        # split the sentence to words
        train_data = []
        if len(data) == 1:
            train_data.extend(data[0].split(self.seperator))
        else:
            [train_data.extend(x.split(self.seperator)) for x in data]

        # get the unique words
        unique_words = list(set(train_data))

        # accoring to unique word to make the data matrix, just random
        word_matrix = np.random.random((len(unique_words), self.n_dim))
        # make the data directory for efficient query
        self.word_vector_dict = dict(zip(unique_words, word_matrix))

        # here is the train array with first cols as right words, second cols with neighbor and negative words,
        # third col is as the truth and wrong label
        train_array = self._get_train_array(data, unique_words)
        train_label = train_array[:, -1].astype(np.float32)

        # now that whole init step has finished, here should start training step
        # loop for the training step, each step, we have to compute the error with truth label
        # and and prediction with logits from logistic function
        for iter in range(self.n_iter):
            # ater get the train array, here I should make the input matrix, output matrix with the
            # already defined word directory
            input_matrix = np.array([self.word_vector_dict[x] for x in train_array[:, 0]])
            output_matrix = np.array([self.word_vector_dict[x] for x in train_array[:, 1]])


            # first to compute the dot product with input and output matrix and get the diagonal value
            diag_array = np.diag(np.dot(input_matrix, output_matrix.T))
            # get the logits
            logits = self._logisic_function(diag_array)

            # compute the error with logits and truth label
            error_array = train_label - logits

            # I have to update the word matrix with the error according to the input matrix add the error
            [self.word_vector_dict[train_array[i, 0]] + error_array[i] for i in range(len(error_array))]


if __name__ == '__main__':
    sen1 = 'hello world I like learning'
    sen2 = 'how are you'
    sen3 = 'this is used to train model'

    sen_list = []
    sen_list.append(sen1)
    sen_list.append(sen2)
    sen_list.append(sen3)

    model = SkipGram()
    model.fit(sen_list)

    # print(model.word_vector_dict)
    res_dict = model.word_vector_dict
    import pandas as pd
    s = pd.Series(res_dict, name='vector')
    s.index.name = 'word'
    df = s.reset_index()
    print(df.head())




