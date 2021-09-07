# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import jieba.analyse as ana
import jieba.posseg as pseg
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
import keras
import matplotlib.pyplot as plt
from matplotlib import style
import time
from gensim.models.word2vec import Word2Vec

style.use('ggplot')

key_size = 256

path = 'F:\working\\201808'
model_path = 'D:\models\\201809'
df = pd.read_excel(path+'/pos_neg.xlsx')
df.head()

# solve the data, drop the nan data, must index the axis
df.drop(['company'], inplace=True, axis=1)
df.dropna(inplace=True)
df.isnull().sum()

# get the title and context
title = df.title
context = df.context
label = df.label

# get the context or cut the title word
def get_keyword(data=context, top=60):
    res = list()
    for i, d in enumerate(data):
        res.append(ana.extract_tags(d, topK=top))
        if i %1000 ==0:
            print('now is step %d'%i)
    return res

def cut_word_with_attr(data=context, allow_pos=['n','nr','nt', 'nz','ns']):
    res = list()
    for i, d in enumerate(data):
        cutted = pseg.cut(d)
        sati = list()
        for word, flag in cutted:
            if flag not in allow_pos:
                continue
            sati.append(word)
        res.append(sati)
        if i % 1000 ==0:
            print('now is step %d'%i)
    return res

keywords = get_keyword(context)

# this uses Doc2Vec algorithm to train the model to get the vector for each document.
def doc2vec(data=keywords, training_epochs=100):
    tagged_data = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(data)]
    # start to train the doc2vec model
    model = Doc2Vec(size=key_size, alpha=.01, min_alpha=.0001, dm=1)

    model.build_vocab(tagged_data)

    epochs = training_epochs
    for i in range(epochs):
        if i % 10 == 0:
            print('now is step %d' % i)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= .0001
        model.min_alpha = model.alpha

    # save the model
    print('Save doc2vec model')
    model.save(model_path+'/doc2vec_keyword20_iter100.bin')

    data = model.docvecs.doctag_syn0
    label = df.label

    return data, label

data, label = doc2vec(keywords)


# this is an algorithm to implement the word2vec also with the TF-IDF algorithm.
# also with the label column, if get the keyword is non, then drop that row.
# return: paragraph vector and new label
def word2vec(data=keywords, use_tfidf=True, label=label):
    def get_word2vec(data=data, min_count=1, iter=100, size=100, workers=5, window=3, sg=1):
        start_time = time.time()
        model = Word2Vec(data, min_count=min_count, iter=iter, size=size, workers=workers, max_vocab_size=None,
                         window=window, sg=sg)
        model_path = 'D:\models\\201809'
        # persist the trained model to disk
        #model.save(model_path + '/word2vec.bin', protocol=2)
        model.save(model_path + '/word2vec.bin')
        # load the trained model
        model = Word2Vec.load(model_path + '/word2vec.bin')
        wordsvec = model[model.wv.vocab]
        uni_words = list(model.wv.vocab)
        print('The word2vec model takes %f seconds' % (time.time() - start_time))
        return wordsvec, uni_words
    wordsvec, uni_words = get_word2vec()
    # make the key-words and word2vec result directory for bellow using the key-value
    res_dic = dict()
    for j in range(len(uni_words)):
        res_dic.update({uni_words[j]: wordsvec[j, :]})

    # make the label columns to be list type. Used in case for the key_word list is null, so just remove the indexed label
    label_list = np.array(label)
    key_words_list = np.array(data)

    # get the null row nums
    null_col = np.empty_like(key_words_list)
    for i in range(len(data)):
        if (len(data[i]) == 0):
            null_col[i] = False
        else:
            null_col[i] = True
    # make the data type to be boolean
    null_col = null_col.astype(np.bool)
    # get the non-null cols
    label_training = label_list[null_col]
    key_words_training = key_words_list[null_col]

    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # tm = TfidfVectorizer().fit(key_words_training).transform(key_words_training)
        key_list = key_words_training.tolist()

        tm = list()
        for i in range(len(key_list)):
            tm.append(" ".join(key_list[i]))
        vec_model = TfidfVectorizer(min_df=1)
        d = vec_model.fit_transform(tm)
        idf = vec_model.idf_
        r_idf = dict(zip(vec_model.get_feature_names(), idf))

        # convert the list_in_array to be a just array
        # compute the tf-idf array for next step to multiply with the each word vector
        conver_array = np.empty_like(key_words_training)
        for i in range(len(key_words_training)):
            tfidf_list = list()
            for j in range(len(key_words_training[i])):
                # first compute the tf value
                tf = key_words_training[i].count(key_words_training[i][j]) / len(key_words_training[i])
                if (key_words_training[i][j] not in r_idf.keys()):
                    # idf = 1.   # change the not exist key-word for 0.0 means that the key-word shows in all corpus log2(n/n) =0.0
                    idf = 0.0
                else:
                    idf = r_idf[key_words_training[i][j]]
                tf_idf = tf * idf
                tfidf_list.extend([tf_idf])
            conver_array[i] = tfidf_list

        # make the key_words sentences vecter by meaning the sum of the all vectors number
        # note we need to substract the null_row_num
        result = np.zeros((key_words_training.shape[0], wordsvec.shape[1]))
        # convert unique list to set is much faster
        res_keys = set(list(res_dic.keys()))
        # loop the all visuable vectors
        for i in range(result.shape[0]):
            for j in range(len(key_words_training[i])):
                # because of the now word2vec model is based on the all satisified word, need to get the satisified key-word vector
                if (key_words_training[i][j] not in res_keys):
                    continue
                # add the tf-idf vector to the res
                result[i, :] = res_dic[key_words_training[i][j]] * conver_array[i][j]
            result[i, :] = result[i, :] / len(key_words_training[i])
        return result, label_training
    else:
        print('TFIDF must be setted True!')

# data, label = word2vec()


# split the data to train and test
xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)


lr = LogisticRegression()
print('baseline accuracy:', lr.fit(xtrain, ytrain).score(xtest, ytest))

from sklearn.svm import SVC
clf = SVC(C=20, kernel='rbf')
print('clf accuracy: ', clf.fit(xtrain, ytrain).score(xtest, ytest))

# use grid search to find best svm C
from sklearn.model_selection import GridSearchCV
params = {'C':[20, 40, 100, 200], 'kernel':['rbf', 'sigmoid']}
grid = GridSearchCV(estimator=clf, param_grid=params)
grid.fit(xtrain, ytrain)
best_clf = grid.best_estimator_
print('best_clf accuracy: ', grid.best_score_)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
print('rfc accuracy: ', rfc.fit(xtrain, ytrain).score(xtest, ytest))





"""
For now, there is  really small datasets, use deep neural networks is not a good idea.
Accuracy is not as good as traditional machine learning algorithm.
"""

# for DNN or LSTM model, label must be categorical
n_classes = len(np.unique(label))
y = keras.utils.to_categorical(label, num_classes=n_classes)
xtrain_new, xtest_new, ytrain_new, ytest_new = train_test_split(data, y, test_size=.2, random_state=1234)

# build a DNN
inputs = Input(shape=[key_size])
dense = Dense(64, activation='relu')(inputs)
dense = BatchNormalization()(dense)
dense = Dropout(.5)(dense)
dense2 = Dense(128, activation='relu')(dense)
dense2 = BatchNormalization()(dense2)
dense2 = Dropout(.5)(dense2)
out = Dense(n_classes, activation='softmax')(dense2)

model = Model(inputs, out)

print('Model structure:')
model.summary()

# compile the model, here is three classes problem
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

es = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)

his = model.fit(xtrain_new, ytrain_new, epochs=100, verbose=1, validation_data=(xtest_new, ytest_new))

def plot_acc(his):
    plt.plot(his.history['acc'], label='train')
    plt.plot(his.history['val_acc'], label='test')
    plt.legend()
    plt.show()
plot_acc(his)

print('DNN model accuracy:', model.evaluate(xtest_new, ytest_new)[1])

# Here I use LSTM to train the model
cnn_kernel_size = int(np.sqrt(key_size))
x = data.reshape(-1, cnn_kernel_size, cnn_kernel_size)
xtrain_l, xtest_l, ytrain_l, ytest_l = train_test_split(x, y, test_size=.2, random_state=1234)

# start to build the LSTM model
inputs = Input(shape=[cnn_kernel_size, cnn_kernel_size])
#lstm = LSTM(64, return_sequences=True, recurrent_dropout=.5)(inputs)
lstm = LSTM(64, recurrent_dropout=.5)(inputs)
lstm = Dropout(.5)(lstm)
dense = Dense(128, activation='relu')(lstm)
out = Dense(n_classes, activation='softmax')(dense)

model = Model(inputs, out)

print('Model Struture:')
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

his = model.fit(xtrain_l, ytrain_l, verbose=1, epochs=100, validation_data=(xtest_l, ytest_l))
plot_acc(his)


