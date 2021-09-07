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
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, Activation
import keras
import matplotlib.pyplot as plt
from matplotlib import style
import copy
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import seaborn as sns


style.use('ggplot')


"""Here to get datasets"""
### load data
path = 'F:\working\\201808'
df = pd.read_excel(path+'/pos_neg.xlsx')

test_df = pd.read_excel(path+'/never_seen.xlsx')

# Because of Doc2Vec is an unsurpervised model, I also load the Hive data for training
hive_df = pd.read_parquet(path + '/respar.gzip')
hive_df.columns = ['title', 'context']
hive_df.dropna(inplace=True)

# solve the data, drop the nan data, must index the axis
df.drop(['company'], inplace=True, axis=1)
df.dropna(inplace=True)
df.isnull().sum()

test_df.drop(['company'], inplace=True, axis=1)

# get the title and context
title = df.title
context_basic = np.array(df.context)

hive_context = np.array(list(set(hive_df.context)))


"""Get keywords or cut words with attribution function"""
# get the context or cut the title word
def get_keyword(data=context_basic, top=40, silent=True, textrank=False):
    res = list()
    for i, d in enumerate(data):
        if textrank:
            res.append(ana.textrank(d, topK=top, allowPOS=['n', 'nr', 'nt', 'nz', 'ns']))
        else:
            res.append(ana.extract_tags(d, topK=top, allowPOS=['n', 'nr', 'nt', 'nz', 'ns']))

        if not silent:
            if data.shape[0] > 10000:
                if i % 10000 == 0:
                    print('now is step %d' % i)
            else:
                if i % 1000 == 0:
                    print('now is step %d' % i)
    return res


def cut_word_with_attr(data=title, allow_pos=['n', 'nr']):
    res = list()
    for i, d in enumerate(data):
        cutted = pseg.cut(d)
        sati = list()
        for word, flag in cutted:
            if flag not in allow_pos:
                continue
            sati.append(word)
        res.append(sati)
        if i % 1000 == 0:
            print('now is step %d' % i)
    return res


"""Get keywords"""
# according to different data source to get different keywords,
# so first we can merge all the keywords to train the doc2vec model,
# then we use the trained model to infer the basic data vector for bellowing classifier training.
#### I have use textRank algorithm to get the keywords, But I get a some worse accuracy for all classifier.
use_extra_data = False
if use_extra_data:
    keywords_hive = get_keyword(hive_context, top=60, silent=False)
keywords_basic = get_keyword(context_basic, top=60, silent=False, textrank=False)

keywords = copy.copy(keywords_basic)
if use_extra_data:
    keywords.extend(keywords_hive)
# Because I find to add data for training the Doc2Vec model doesn't give a better accuracy for SVM,
# so change to original datasets.
keywords = keywords_basic

tagged_data = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(keywords)]


"""Use Doc2Vec algorithm to train keywords list"""
# start to train the doc2vec model
def doc2vec_basic(epochs=200, dm=1, vector_size=100, silence=True, min_count=3, alpha=.03, tagged_data=tagged_data):
    model = Doc2Vec(vector_size=vector_size, alpha=.03, dm=dm, window=2, min_count=min_count)

    model.build_vocab(tagged_data)

    epochs = epochs
    for i in range(epochs):
        if not silence:
            if vector_size > 100:
                if i % 20 == 0:
                    print('now is step %d'%i)
                    model.alpha -= .002   # change the alpha with : alpha - 0.002*epochs/20
                    model.min_alpha = model.alpha
            else:
                # if the needed vector size is small, then just change the alpha every 2 epochs
                if i % 2 == 0:
                    model.alpha -= .002
                    model.min_alpha = model.alpha     # fix the change learning rate.
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        # model.min_alpha = model.alpha
    return model

def doc2vec(dm=0):
    model = Doc2Vec(vector_size=100, alpha=.03, min_alpha=.00001, dm=dm, window=2, min_count=2)
    model.build_vocab(tagged_data)
    # not use for loop to train the model, just change the epochs parameter
    model.train(tagged_data, total_examples=model.corpus_count, epochs=500)
    print('Finished training doc2vec model!')
    return model

# according to the comparasion for different epochs and different algorithms,
# I find that for 10 epochs and dm=0 and dimension is 70 get best accuracy
# For with no learning rate decay get best.
vector_size_before = 100
m_count_before = 5
lr_before = .025

vector_size = 100
m_count = 7
lr = .020
model = doc2vec_basic(10, 0, vector_size_before, min_count=m_count_before, alpha=lr_before)

# Because I add the training data, so not get all vector for training, here just get trained model
#data = model.docvecs.vectors_docs



# After training doc2vec model finished, I have to infer the basic also known as needed training data to vector

# data = np.empty((len(keywords_basic),  70))
# for i in range(len(data)):
#     data[i, :] = model.infer_vector(keywords_basic[i])


#model.random.seed(0)
data = model.docvecs.vectors_docs[:len(keywords_basic), :]
label = df.label

### whether I use alreadly trained model to infer the wantted training vector will be better? just try
### I have tried, not good as before....
# model.random.seed(0)
# data = np.empty((len(keywords_basic), vector_size))
# for i in range(len(keywords_basic)):
#     data[i, :] = model.infer_vector(keywords_basic[i])


"""Compare different parameters to effect model accuracy, plot result"""
xtrain, xtest,  ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

# this is used to compare with different epochs and mds for SVM accuracy.
def compare_differet_epochs_acc():
    epochs = [10, 20]
    dm = [0]
    vector_size = [70, 100]
    min_count = [3, 5, 7]
    lr = [.020, .025, .03]
    res = dict()
    for ep in epochs:
        for r in dm:
            for v in vector_size:
                for m in min_count:
                    for l in lr:
                        model_tmp = doc2vec_basic(ep, r, v, silence=True, alpha=l)
                        data_tmp = model_tmp.docvecs.vectors_docs[:len(keywords_basic), :]
                        xtrain_tmp, xtest_tmp, ytrain_tmp, ytest_tmp = train_test_split(data_tmp, label, test_size=.2,
                                                                                        random_state=1234)
                        svm = SVC(C=20, kernel='rbf')
                        svm.fit(xtrain_tmp, ytrain_tmp)
                        res[str(ep) + '_' + str(r) + '_vector' + str(v) + '_mincount' + str(m) + 'lr_' + str(
                            l)] = svm.score(xtest_tmp, ytest_tmp)
    print('Finished all model training! ')
    plt.plot(res.values())
    plt.xticks(np.arange(len(res)), res.keys(), rotation=90)
    plt.title('All model accuracy')
    plt.show()

compare_differet_epochs_acc()


# get the test data keyword, and get the index infer vector to test.
test_context = test_df.context
test_label = test_df.label

test_keywords = get_keyword(test_context)

# use the doc2vec model to infer the test data vector
test_vector = np.empty((len(test_keywords), data.shape[1]))
for t in range(len(test_keywords)):
    test_vector[t, :] = model.infer_vector(test_keywords[t], steps=10, alpha=.03)


"""Here is machine learning algorithms"""
lr = LogisticRegression(C=10, fit_intercept=True)
print('baseline accuracy:', lr.fit(xtrain, ytrain).score(xtest, ytest))

from sklearn.svm import SVC
clf = SVC(C=20, kernel='rbf', probability=True)
print('support vector machine accuracy: ', clf.fit(xtrain, ytrain).score(xtest, ytest))


"""Here is for plot parameters cross-validation box curve"""
### This function is just used to plot different algorithm with different parameters  CV score boxploting.
def plot_diff_box(estimator, x, y, params=None, cv=10, cut_off=True, cut_range=None, metric=None):
    scores = list()
    inx = list()
    estimator = estimator
    x, y = x, y
    if params is None and not cut_off:
        return 'Must be using for estimator choosen or for different cutoff for probability!'
    if cut_off:
        if len(np.unique(y)) != 2:
            return 'Cut_off is must for Binary!'

        # this is for different threshold for predicting.
        def _cut_off(estimator, x, c):
            return (estimator.predict_proba(x)[:, 1] > c).astype(int)

        def _score_f(c):
            def _s(estimator, x, y):
                if metric is None:  # just use accuracy for evaluating
                    return metrics.accuracy_score(y, _cut_off(estimator, x, c))
                else:
                    return metric(y, _cut_off(estimator, x, c))

            return _s

        if cut_range is None:  # if range is None, then use 0.1-0.9 every 0.1 range
            cut_range = np.arange(0.1, 0.9, 0.1)

        # loop for cut_off
        for c in cut_range:
            scores.append(cross_val_score(estimator, x, y, cv=cv, scoring=_score_f(c)))

        # loop for the inx
        for i in range(len(cut_range)):
            for _ in range(cv):
                inx.append(round(cut_range[i], 2))
    else:  # if not for cut_off, then must be parameter choosen
        if params is None:
            return 'If not for Cut_off, then must provide params. '

        # loop for params
        for p in params:
            svm.C = p
            scores.append(cross_val_score(estimator, x, y, cv=cv))

        # also for using inx
        for i in range(len(params)):
            for _ in range(cv):
                inx.append(params[i])

    s = np.array(scores).reshape(-1, 1)

    s_p = pd.DataFrame(s)
    indx_p = pd.DataFrame(np.array(inx))
    t_df = pd.concat((indx_p, s_p), axis=1)
    t_df.columns = ['inx', 'score']

    # now plot result
    sns.boxplot(x=t_df.inx, y=t_df.score)
    plt.title('Different C box curve')
    plt.show()


svm = SVC()
plot_diff_box(svm, xtrain, np.array(ytrain), params=[10, 30, 50, 70, 100], cut_off=False)



"""This is ensemble algorithms"""
#### Because for now, I just use the single algrithm to train the model, why not use Ensemble algorithm to
# combine all the classifier with different weights to find whether or not I can get better accuracy on Test Data.

# first use is bagging.
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':[30, 50, 100]}
bagging_clf = BaggingClassifier(base_estimator=clf, n_estimators=20, random_state=1234, max_features=.5, max_samples=.5)
grid_bagging = GridSearchCV(estimator=bagging_clf, param_grid=params, cv=3)

grid_bagging.fit(xtrain, ytrain)

print('Bagging SVM accuracy: ', grid_bagging.score(xtest, ytest))


# second I use voting algorithm to vote to get final result,
# Here I select SVM, LR, GradientBoostingTree as base
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
estimator = [('lr', lr), ('svm', clf), ('gbc', gbc)]

## Here I use soft voting to get the prediction
eclf = VotingClassifier(estimators=estimator, voting='soft', weights=[1, 10, 1])

# params_eclf = {'weights':[[1, 10, 1], [1, 15, 3], [2, 10, 5]]}

# grid_eclf = GridSearchCV(estimator=eclf, param_grid=params_eclf)
# grid_eclf.fit(xtrain, ytrain)

# print('Best eclf acc:', grid_eclf.score(xtest, ytest))

eclf.fit(xtrain, ytrain)

print('Voting Algorithm Accuracy:', eclf.score(xtest, ytest))
### Best Accuracy: 0.8517350157728707, ('lr', lr), ('svm', clf), ('gbc', gbc), weights=[1, 10, 1]
"""Best Accuracy:0.8517350157728707, """

### Above I use voting equals soft to train the model, Here I use the Voting to 'hard' means majority voting.

eclf_hard = VotingClassifier(estimators=estimator, voting='hard')

eclf_hard.fit(xtrain, ytrain)

print('Hard Voting ACC:', eclf_hard.score(xtest, ytest))
""" Not good as 'soft', it's 0.8115141955835962"""



"""Model parameters grid search"""
# use grid search to find best svm C
from sklearn.model_selection import GridSearchCV
params = {'C':[1, 20, 40], 'kernel':['rbf']}
grid = GridSearchCV(estimator=clf, param_grid=params)
grid.fit(xtrain, ytrain)
best_clf = grid.best_estimator_
print('best_clf accuracy: ', grid.best_score_)
print('best clf test accuracy :', best_clf.score(xtest, ytest))
print('best svm: ', best_clf)



"""LightGBM algorithm"""
### This is used lightGBM for model training
import lightgbm as lgb
from sklearn import metrics


def lgb_model():
    # because of lightgbm label must be between 0-3, first convert the label
    label_lgb = copy.copy(label)
    label_lgb[label_lgb == -1] = 2
    xtrain_lgb, xtest_lgb, ytrain_lgb, ytest_lgb = train_test_split(data, label_lgb, test_size=.2, random_state=1234)
    # training parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'accuracy',
        'num_leaves': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': 1
    }

    # training datasets
    lgb_train = lgb.Dataset(xtrain_lgb, ytrain_lgb)
    lgb_test = lgb.Dataset(xtest_lgb, ytest_lgb, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_train)

    lgb_pred = np.argmax(gbm.predict(xtest), axis=1)

    print('LightGBM accuracy : ', metrics.accuracy_score(ytest_lgb, lgb_pred))


lgb_model()

# print('LightGBM test accuracy: ', metrics.accuracy_score(ytest, lgb_pred))



"""Model confusion matrix"""
from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(ytest, clf.predict(xtest))
plt.show()



"""Gradient Boosting Tree grid search"""
# grid search for gradient boosting tree
params_gbc = {'max_depth':[5, 7], 'n_estimators':[100, 150, 200]}
grid_gbc = GridSearchCV(gbc, param_grid=params_gbc, cv=3)
grid_gbc.fit(xtrain, ytrain)



"""Convert 3-classes to 2-classes"""
import copy
from scikitplot.metrics import plot_roc_curve
label_new = copy.copy(label)
label_new[label_new == -1] = 1



"""Plot confusion matrix and ROC curve"""
# split to train and test data
xtrain_new, xtest_new, ytrain_new, ytest_new = train_test_split(data, label_new, test_size=.2, random_state=1234)

lr.fit(xtrain_new, ytrain_new)
print('Basic LR: ', lr.score(xtest_new, ytest_new))
plot_confusion_matrix(ytest_new, lr.predict(xtest_new))
plot_roc_curve(ytest_new, lr.predict_proba(xtest_new))
plt.show()



"""Bellow is Deep neural Nets"""
# for DNN or LSTM model, label must be categorical
y = keras.utils.to_categorical(label_new, num_classes=2)
xtrain_new, xtest_new, ytrain_new, ytest_new = train_test_split(data, y, test_size=.2, random_state=1234)
es = keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)

def plot_acc(his, title='Train and Test Data Accuracy'):
    plt.plot(his.history['acc'], label='train')
    plt.plot(his.history['val_acc'], label='test')
    plt.title(title)
    plt.legend()
    plt.show()



"""This is DNN model"""
y_mul = keras.utils.to_categorical(label, num_classes=3)
xtrain_mul, xtest_mul, ytrain_mul, ytest_mul = train_test_split(data, y_mul, test_size=.2, random_state=1234)


def dnn_net(data=None, label=None, binary=True, epochs=100, hidden_units=128, xtrain=xtrain_new, xtest=xtest_new,
            ytrain=ytrain_new, ytest=ytest_new):
    inputs = Input(shape=[xtrain.shape[1]])
    dense = Dense(hidden_units)(inputs)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(.5)(dense)
    dense2 = Dense(hidden_units)(dense)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(.5)(dense2)
    if binary:
        out = Dense(2, activation='sigmoid')(dense2)
    else:
        out = Dense(3, activation='softmax')(dense2)

    model = Model(inputs, out)

    print('Model structure:')
    model.summary()

    if data is not None and label is not None:
        xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

    # compile the model, here is three classes problem
    if binary:
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        his = model.fit(xtrain, ytrain, epochs=epochs, verbose=1, validation_data=(xtest, ytest))
        plot_acc(his, title='Binary Accuracy')
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_new, ytest_new)[1])
    else:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        his = model.fit(xtrain, ytrain, epochs=epochs, verbose=1, validation_data=(xtest, ytest))
        plot_acc(his, title='Multi-class Accuracy')
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_mul, ytest_mul)[1])

    print('All Finished !')
    return model
# This is multi-class DNN model
dnn_net(binary=False, epochs=150, hidden_units=50, xtrain=xtrain_mul, xtest=xtest_mul, ytrain=ytrain_mul, ytest=ytest_mul)
# this is Binary class for DNN model
dnn_net(binary=True, epochs=100, hidden_units=50)
print('DNN model accuracy:', model.evaluate(xtest_new, ytest_new)[1])



"""Here is LSTM model"""
# Here I use LSTM to train the model
x = data.reshape(-1, 10, 10)
xtrain_l, xtest_l, ytrain_l, ytest_l = train_test_split(x, label, test_size=.2, random_state=1234)
# start to build the LSTM model
inputs = Input(shape=[10, 10])
lstm = LSTM(64, return_sequences=True, recurrent_dropout=.5)(inputs)
lstm = LSTM(64, recurrent_dropout=.5)(lstm)
dense = Dense(128, activation='relu')(lstm)
out = Dense(3, activation='softmax')(dense)

model = Model(inputs, out)

print('Model Struture:')
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
his = model.fit(xtrain_l, ytrain_l, verbose=1, epochs=100, validation_data=(xtest_l, ytest_l))
plot_acc(his)



"""This is data visualization"""
### This is for plotting the training data of different classes. Maybe can find how difficult to
# train the classifier to separate the datasets.
# use PCA to decomposition the training data
# then plot the scatter curve to find whether training result is good
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(xtrain)
training_pca = pca.transform(xtrain)
# label_pca = copy.copy(np.array(ytrain))
# label_pca[label_pca == -1] = 2
label_pca = ytrain

#  loop the training data appended with the label
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for i in np.unique(label_pca):
#     if i == 1:
#         continue
    ax.scatter(training_pca[label_pca==i, 0], training_pca[label_pca==i, 1], label=str(i))
plt.title('cluster for all training data')
plt.legend()
plt.show()


# up is use PCA to plot the result, here I use T-SNE to plot the result data to find something important.
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, )
tsne_data = tsne.fit_transform(data)
tsne_label = label_new

# plot the result of TSNE
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for i in np.unique(tsne_label):
#     if i == -1:
#         continue
    plt.scatter(tsne_data[tsne_label == i, 0], tsne_data[tsne_label == i, 1], label='class_'+ str(i))
plt.title('T-SNE curve')
plt.legend()
plt.show()


## Because I have the data's label, I want to do clustering for these datasets. Just want to make the cluster to be accurate.
# If so, I can use this clustering model to make the unknown data tagged with label.
## this function is used to plot the data with labels using PCA to decompsite data
from sklearn.decomposition import PCA


def plot_high_dimension_data(data, label, title='Prediction', plot_classes=None):
    pca = PCA(n_components=2)
    data_new = pca.fit(data).transform(data)
    uni_label = np.unique(label)

    # for loop the data with unique label
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if plot_classes is not None:
        for p_c in plot_classes:
            if p_c not in uni_label:
                return 'Wanted plot classes %s not in label.' % (str(p_c))
    else:
        plot_classes = uni_label

    for l in plot_classes:
        ax.scatter(data_new[label == l, 0], data_new[label == l, 1], label='class_' + str(l))

    plt.legend()
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

cluster_res = kmeans.predict(data)
# print('Cluster center is :', kmeans.cluster_centers_)
## Plot the result
plot_high_dimension_data(data, cluster_res)
plot_high_dimension_data(data, label, title='True')



"""Here is CNN model"""
#### Because of the tensorflow-hub can not be used to train the model, Here I build a CNN model to train the model
### Because for CNN and LSTM, I have to make the data to be 3-D, Here it is.
label_binary = copy.copy(label)
label_binary[label_binary==-1] = 1

y_third = keras.utils.to_categorical(label, num_classes=3)
y_binary = keras.utils.to_categorical(label_binary, num_classes=2)

# make data to be 3-D
data_deep = data.reshape(-1, 10, 10)

# split the binary and third classes to train and test data
xtrain_third, xtest_third, ytrain_third, ytest_third = train_test_split(data_deep, y_third, test_size=.2, random_state=1234)
xtrain_binary, xtest_binary, ytrain_binary, ytest_binary = train_test_split(data_deep, y_binary, test_size=.2, random_state=1234)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D, Flatten


def cnn_basic(binary=True, epochs=100):
    inputs = Input(shape=(10, 10))

    conv1 = Conv1D(64, 1, padding='SAME')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(.5)(conv1)

    conv2 = Conv1D(64, 1, padding='SAME')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(.5)(conv2)

    # flatten the learned features and add dense layer
    fc1 = Flatten()(conv2)
    fc1 = Dense(64)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(.5)(fc1)

    # compile the model, here is three classes problem
    if binary:
        out = Dense(2, activation='sigmoid')(fc1)
        model = Model(inputs, out)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        print('Model structure:')
        model.summary()
        his = model.fit(xtrain_binary, ytrain_binary, epochs=epochs, verbose=1,
                        validation_data=(xtest_binary, ytest_binary))
        plot_acc(his)
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_binary, ytest_binary)[1])
    else:
        out = Dense(3, activation='softmax')(fc1)
        model = Model(inputs, out)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        print('Model structure:')
        model.summary()
        his = model.fit(xtrain_third, ytrain_third, epochs=epochs, verbose=1,
                        validation_data=(xtest_third, ytest_third))
        plot_acc(his)
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_third, ytest_third)[1])

# this is multi-class model training result.
cnn_basic(binary=False)
# this is for binary problem result.
cnn_basic(binary=True)



"""Here is Residual network"""
def residual_network_basic(binary=True, n_layer=4, flatten=True, epochs=100, hidden_units=128):
    inputs = Input(shape=(10, 10))

    def _res_block(layer):
        res = Conv1D(10, 1, padding='SAME')(layer)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Dropout(.5)(res)

        res = Conv1D(10, 1, padding='SAME')(res)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Dropout(.5)(res)

        return keras.layers.add([layer, res])

    # build the residual layer
    for i in range(n_layer):
        if i == 0:
            res = _res_block(inputs)
        else:
            res = _res_block(res)

    # after constructing the block layer, then use the global average pooling or flatten the result
    if flatten:
        res = Flatten()(res)
    else:
        res = GlobalAveragePooling1D()(res)

    # Here just append with one dense layer, also with batchnormalization
    res = Dense(hidden_units)(res)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    res = Dropout(.5)(res)

    # for different problem to train different model
    if binary:
        out = Dense(2, activation='sigmoid')(res)
        model = Model(inputs, out)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        print('Model structure:')
        model.summary()
        his = model.fit(xtrain_binary, ytrain_binary, epochs=epochs, verbose=1,
                        validation_data=(xtest_binary, ytest_binary))
        plot_acc(his, title='Binary train and test Accuracy')
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_binary, ytest_binary)[1])
    else:
        out = Dense(3, activation='softmax')(res)
        model = Model(inputs, out)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        print('Model structure:')
        model.summary()
        his = model.fit(xtrain_third, ytrain_third, epochs=epochs, verbose=1,
                        validation_data=(xtest_third, ytest_third))
        plot_acc(his, title='Multi-class train and test Accuracy')
        print('After training, Model Test Sets Accuracy : ', model.evaluate(xtest_third, ytest_third)[1])

### this is binary problem for residual network
residual_network_basic(binary=True, n_layer=4)
### this is for multi-class of residual network
residual_network_basic(binary=False, n_layer=5, epochs=100)



"""Here is Dense Net"""
def Dense_net(data=None, label=None, binary=False, epochs=100, n_layers=4, conv_units=64, flatten=True, dense_units=128,
              batch_size=512,
              xtrain=xtrain_binary, xtest=xtest_binary, ytrain=ytrain_binary, ytest=ytest_binary):
    if data is not None:
        input_dim_1 = data.shape[1]
        input_dim_2 = data.shape[2]
    if data is None and xtrain is not None:
        input_dim_1 = xtrain.shape[1]
        input_dim_2 = xtrain.shape[2]

    inputs = Input(shape=(input_dim_1, input_dim_2))

    # this is dense net residual block.
    def _res_block(layers=inputs, added_layer=inputs):
        res = Conv1D(conv_units, 1, padding='SAME')(layers)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Dropout(.5)(res)

        res = Conv1D(input_dim_2, 1, padding='SAME')(res)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Dropout(.5)(res)

        return keras.layers.add([res, added_layer])

    for i in range(n_layers):
        if i == 0:
            res = _res_block()
        else:
            res = _res_block(layers=res)

    if flatten:
        res = Flatten()(res)
    else:
        res = GlobalAveragePooling1D()(res)

    res = Dense(dense_units)(res)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    res = Dropout(.5)(res)

    if data is not None and label is not None:
        xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

    if binary:
        out = Dense(2, activation='sigmoid')(res)
        model = Model(inputs, out)
        print('Model Structure:')
        model.summary()
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        his = model.fit(xtrain, ytrain, epochs=epochs, verbose=1, validation_data=(xtest, ytest), batch_size=batch_size)
        plot_acc(his)
        print('Binary model Test Accraucy:', model.evaluate(xtest_binary, ytest_binary)[1])
    else:
        if len(ytrain.shape) == 2 and ytrain.shape[1] != 3:
            return 'For multi-classes, ytrain and ytest must be provived!'

        out = Dense(3, activation='softmax')(res)
        model = Model(inputs, out)
        print('Model Summary:')
        model.summary()
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        his = model.fit(xtrain, ytrain, epochs=epochs, verbose=1, validation_data=(xtest, ytest), batch_size=batch_size)
        plot_acc(his)
        print('Multi-class model Test Accuracy: ', model.evaluate(xtest_third, ytest_third)[1])
    print('ALL Finished!')
    return model

### This is Binary
dense_model_binary = Dense_net(binary=True, n_layers=3, flatten=True)
dense_model_multi = Dense_net(binary=False)



"""Use ES data"""
### Here I use elasticsearch to read ipin data, with content as data and sentiment as label.
# I just want to add more and more data for model training. Also want to get better result. HA
from elasticsearch import Elasticsearch

content = list()
sentiment = list()

es = Elasticsearch(hosts='192.168.1.34', db=0)
doc = {'size':10000, 'query':{'match_all':{}}}

index = 'nmas'
doc_type = 'public_opinion_result'

# for now, there is just about 13k datasets.
res1 = es.search(index=index, doc_type=doc_type, body=doc, scroll='1m')
scroll = res1['_scroll_id']
res2 = es.scroll(scroll_id=scroll, scroll='1m')

res1_data = res1['hits']['hits']
res2_data = res2['hits']['hits']

content_list = list()
sentiment_list = list()

for i in range(len(res1_data)):
    op_content = res1_data[i]['_source']['opinion_content']
    op_sen = res1_data[i]['_source']['opinion_sentiment']
    if op_content is None:
        content_list.append('Nan')
    if op_sen is None:
        sentiment_list.append('Nan')
    content_list.append(op_content)
    sentiment_list.append(op_sen)

for j in range(len(res2_data)):
    op_content = res1_data[j]['_source']['opinion_content']
    op_sen = res1_data[j]['_source']['opinion_sentiment']
    if op_content is None:
        content_list.append('Nan')
    if op_sen is None:
        sentiment_list.append('Nan')
    content_list.append(op_content)
    sentiment_list.append(op_sen)



"""Make ES data combined with original data to train model"""
### just to find how much label I get from the es
import seaborn as sns
from collections import Counter
# a = pd.DataFrame(label_es_sent)
print('This is es label counted:',Counter(sentiment_list))
print('This is original label counted: ',Counter(label))

# convert the getted es result, and combine them with context_basic data.
context_es = np.array(content_list)
label_es_sent = np.array(sentiment_list)

# this is combined context.
context_added_es = np.concatenate((context_basic, context_es), axis=0)
label_added_es = np.concatenate((np.array(label), label_es_sent), axis=0)
print('Now here is %d samples'%(context_added_es.shape[0]))

### Before I use the data, first dump the data into disk.
context_es_pd = pd.DataFrame(context_es.reshape(-1, 1))
label_es_pd = pd.DataFrame(label_es_sent.reshape(-1, 1))

es_pd = pd.concat((context_es_pd, label_es_pd), axis=1)
es_pd.columns = ['context', 'label']

# dump data to disk
es_pd.to_excel(path+'/es_data.xlsx', encoding='gb180030', index=False)
# get the keywords for new datasets
keywords_added_es = get_keyword(context_added_es, top=60, silent=True)
# after I get keywords, I also need to do Doc2Vec model training. Here is just same as Before.
tagged_data_es = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(keywords_added_es)]
vector_size_before = 100
m_count_before = 5
lr_before = .025

vector_size = 100
m_count = 7
lr = .020
model = doc2vec_basic(10, 0, vector_size_before, min_count=m_count_before, alpha=lr_before, tagged_data=tagged_data_es)

# After I get already trained doc2vec model, here I just get all trained vector for machine learning model training.
data_es = model.docvecs.vectors_docs[:len(keywords_added_es), :]

# because original label is 3-classes: -1, 0, 1, convert to binary classes
label_es = label_added_es
label_es[label_es == -1] = 1



"""Here is Word2vec combined with TFIDF"""
##### Because according to my previous job for using the Doc2vec model, for adding new datasets,
# I can not get a better representation for all documents. So for bellowing function
# just uses extracted keywords for each paragraph, and use the Word2Vec model to train the each word, also with TFIDF weighted.

from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import time

word2vec_size = 100


### this is used for get IDF module function
def get_idf(data=keywords_added_es):
    # make all each getted keywords not use list as split, use blank as split for using tfidf
    tm = list()
    for i in range(len(data)):
        tm.append(' '.join(data[i]))

    idf_model = TfidfVectorizer(min_df=1)
    idf = idf_model.fit(tm).idf_
    idf_dict = dict(zip(idf_model.get_feature_names(), idf))

    return idf_dict


### this is final tf-idf matrix result, multiply with word2vec result
def tfidf(idf_dict, word_dict, data=keywords_added_es, word2vec_size=word2vec_size, topk=60):
    # Here I want to make tfidf result to be n*topk*word2vec_size, if word in idf-dictory, then all 3-D will be just same as tfidf
    result = np.empty((len(data), topk, word2vec_size), dtype=np.float32)

    # returned matrix is must be same like 'data', first compute each sentence 'tf' value, then multiply with getted words 'idf' value
    #     tfidf = np.empty_like(np.array(data))
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

### this is a function for get the basic word2vec result matrix
def get_basic_word2vec_matrix(word_dict, data=keywords_added_es, word2vec_size=word2vec_size, topk=60):
    print(len(keywords_added_es))
    res = np.empty((len(data), topk, word2vec_size), dtype=np.float32)

    words_key = word_dict.keys()

    # for loop
    for r in range(len(data)):
        for r2 in range(len(data[r])):
            if data[r][r2] in words_key:
                res[r][r2][:] = word_dict[data[r][r2]]
            else:
                res[r][r2][:] = 0

    return res

def word2vec_tf(data=keywords_added_es, use_tfidf=False, min_count=1, iter=20, size=word2vec_size, workers=5, window=3,
                sg=1, save_model=False):
    s_t = time.time()

    print('Now is training for word2vec model')
    model = Word2Vec(data, min_count=min_count, iter=iter, size=size, workers=workers, window=window, sg=1,
                     max_vocab_size=None)
    if save_model:
        model.save(path + '/tmp_model/word2vec.bin', protocol=2)

    wordsvec = model[model.wv.vocab]
    uni_words = list(model.wv.vocab)
    # construct a dict for word2vec result
    word_dict = dict()
    for m in range(len(uni_words)):
        word_dict[uni_words[m]] = wordsvec[m, :]

    if use_tfidf:
        print('Now is using tfidf')
        # this is IDF dictory
        idf_dict = get_idf(data)
        # this result is n*60*100
        result_returned = tfidf(idf_dict, word_dict, data)
    else:
        print('This is not using tfidf')
        # if do not use tfidf, then just make result to be 3-D, for just getted result
        result_returned = get_basic_word2vec_matrix(word_dict, data)

    e_t = time.time()
    if use_tfidf:
        print('Using TFIDF, get final result use {:.4f} seconds'.format((e_t - s_t)))
    else:
        print('Not using Tfidf, GET word2vec result use {:.4f} seconds'.format((e_t - s_t)))

    return result_returned
result = word2vec_tf(use_tfidf=True)


"""Here is residual network for adding ES data model"""
"""This class implements DenseNet, also can use it as basic residual networks, use parameter 'basic_residual' to choose
    whether or not to use residual networks.
"""

import keras
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Dense, Dropout, Activation, GlobalAveragePooling1D, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

class denseNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, basic_residual=False, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy'):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_classes = n_classes
        self.basic_residual = basic_residual
        self.n_layers = n_layers
        self.flatten = flatten
        self.use_dense = use_dense
        self.n_dense_layers = n_dense_layers
        self.conv_units = conv_units
        self.stride = stride
        self.padding = padding
        self.dense_units = dense_units
        self.drop_ratio = drop_ratio
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = self._init_model()

    # this will build DenseNet or ResidualNet structure, this model is already compiled.
    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers, added_layers=inputs):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            if self.basic_residual:
                return keras.layers.add([res, layers])
            else:
                return keras.layers.add([res, added_layers])

        # construct residual block chain.
        for i in range(self.n_layers):
            if i == 0:
                res = _res_block(inputs)
            else:
                res = _res_block(res)

        # using flatten or global average pooling to process Convolution result
        if self.flatten:
            res = Flatten()(res)
        else:
            res = GlobalAveragePooling1D()(res)

        # whether or not use dense net, also with how many layers to use
        if self.use_dense:
            for j in range(self.n_dense_layers):
                res = Dense(self.dense_units)(res)
                res = BatchNormalization()(res)
                res = Activation('relu')(res)
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_classes must up to 2!')

        return model

    # Fit on given training data and label. Here I will auto random split the data to train and validation data,
    # for test datasets, I will just use it if model already trained then I will evaluate the model.
    def fit(self, data, label, epochs=100, batch_size=256):
        # self.model = self._init_model()
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                             validation_data=(xvalidate, yvalidate), batch_size=batch_size)
        print('After training, model accuracy on validation datasets is {:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        return self

    # evaluate model on test datasets.
    def evaluate(self, data, label, batch_size=None, silent=False):
        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.6f}'.format(acc))
        return acc

    # plot after training accuracy and loss curve.
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['acc'], label='Train Accuracy')
        ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()
# first convert the label to be 3-D
label_es_deep_mul = keras.utils.to_categorical(label_es, num_classes=3)
xtrain_tf, xtest_tf, ytrain_tf, ytest_tf = train_test_split(result, label_es_deep_mul, test_size=.2, random_state=1234)

dense_model = denseNet(input_dim1=result.shape[1], input_dim2=result.shape[2], n_classes=3, n_layers=2, n_dense_layers=1,
                       use_dense=False, flatten=False)
dense_model.fit(xtrain_tf, ytrain_tf, batch_size=256)
"""Model evaluation on validation datasets accuracy:0.7451"""