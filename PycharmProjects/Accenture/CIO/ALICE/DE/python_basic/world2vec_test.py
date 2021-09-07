# -*- coding:utf-8 -*-
"""This is to implement just the a random world2vec model that just based on some small vocabulary"""
import numpy as np
import random

ndim = 5   # matrix dimension
slice_window = 3  # slide windows
n_negative = 2  # add how many negative words

# first make some sentence
sen1 = 'hello world I like learning'
sen2 = 'how are you'
sen3 = 'this is used to train model'

sen_list = []
sen_list.append(sen1)
sen_list.append(sen2)
sen_list.append(sen3)

# first to make the one-hot encoder for whole vectors, get unique world
unique_word = []
for s in sen_list:
    unique_word.extend(s.split(' '))
unique_word = list(set(unique_word))


# make the random matrix with the unique word
word_matrix = np.random.random((len(unique_word), ndim))

# make the training dictory for later step
word_dict = {}
for k, v in zip(unique_word, word_matrix):
    word_dict[k] = v

print(word_dict)

# accorring to skip-gram model, buid the training data
train_data = []
# loop for the whole sentence to make the training data
for s in sen_list:
    s_split = s.split(' ')
    if len(s_split) < slice_window:
        train_data.append([s_split[-1], s_split[-2], 1])
    else:   # if sentence too small, don't add negative words
        for k in range(len(s_split) - slice_window + 1):  # loop for the whole word
            train_data.append([s_split[k + int(slice_window / 2)], s_split[k + int(slice_window/2) - 1], 1])
            train_data.append([s_split[k + int(slice_window / 2)], s_split[k + int(slice_window / 2) + 1], 1])

            # here I add some negative sampling data
            # for every slide, add some negative words, random choose some words not in the sentence
            # get the not appear word list in this sentence
            not_appear_words = list(set(unique_word) - set(s_split))
            train_data.extend([s_split[k + int(slice_window / 2)], rand_word, 0]
                              for rand_word in random.choices(not_appear_words, k=n_negative))

train_data = np.array(train_data)

# after I have get the training data, here I should make the input matrix and output matrix, also with label
input_matrix = np.empty((len(train_data), ndim))
output_matrix = np.empty_like(input_matrix)
label_matrix = np.empty((len(train_data), ))

# loop for the whole train data
for i in range(len(train_data)):
    input_matrix[i, :] = word_dict[train_data[i, 0]]
    output_matrix[i, :] = word_dict[train_data[i, 1]]
    label_matrix[i] = train_data[i, 2]

# after I have get with the input and output matrix, I could get the dot product with prediction
pred = np.diag(np.dot(input_matrix, output_matrix.T) * np.eye(len(input_matrix)))

# here I have to write the logistic regression function to turn the prediction to prob
def logisic_function(x):
    return 1. / (1 + np.exp(-x))

pred_prob = logisic_function(pred)

# compute the error with prediction and label
error = label_matrix - pred_prob

# adjust with the word matrix with error, before do that, here are with the input matrix word order
# to adjust the word matrix
for i in range(len(error)):
    word_dict[train_data[i, 0]] += error[i]

print(word_dict)



