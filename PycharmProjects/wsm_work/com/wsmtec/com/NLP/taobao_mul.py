# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
import jieba
import jieba.analyse as ana
import jieba.posseg as pseg
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import *
import logging
import keras
import tensorflow as tf
import time
from itertools import chain
# build the cnn to train the model
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM, Embedding, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Activation, UpSampling1D, Flatten
from keras.optimizers import SGD
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_custom_objects
from keras import backend as k

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
mpl.rcParams['font.sans-serif'] = ['SimHei']

# set the jieba new directory
new_dic_path = 'F:\workingData\\201804\\taobao\dict_new.txt'
jieba.load_userdict(new_dic_path)

connection = pymysql.connect(user='zhanghui',password='zhanghui',database='model_data',host='10.1.36.18',charset='utf8')
# query = 'select item_name,label as label from model_data.tb_item_label_cfm2 where label is not null and sorttime <= \'2018-03-01\' '
query = 'select * from model_data.train_data_import_db'
re = pd.read_sql(query,con=connection)

from sklearn.preprocessing import LabelEncoder
le_model = LabelEncoder().fit(re['label'])
new_label = le_model.transform(re['label']).astype(np.int32)
items = np.array(re['item_name'])
label = np.array(new_label)

# get the random index for training, because that too much data, the model training time is really long.
p = 1
rand_choice = np.random.choice(a=[True, False], size=(items.shape[0]), p=(p,1-p ))
print('the sum of all sati data is ', np.sum(rand_choice))
item_choice = items[rand_choice]
label_choice = label[rand_choice]

def cut_words(data=item_choice,cut_all=False):
    cut_words_list = list()
    for i, d in enumerate(data):
        cut_words_list.append(list(jieba.cut(d, cut_all=cut_all)))
        if(i % 20000 == 0 ):
            print('Now is document %d'%(i))
            break
    return cut_words_list

# according to attribute of the word, get the satisified word
def cut_words_with_attr(data=item_choice, pos_allow=True, allow_pos=['n','nr']):
    return_sati_list = list()
    for i, d in enumerate(data):
        cut_data = pseg.cut(d)
        sati_sati = list()
        for word, flag in cut_data:
            if pos_allow:
                if(flag not in allow_pos):
                    continue
            sati_sati.append(word)
        return_sati_list.append(sati_sati)
        if(i % 20000 == 0 ):
            print('Now is step %d'%i)
    return return_sati_list
cut_words_with_attr_list = cut_words_with_attr()

""" cut each sentence to be separated words """
def jieba_cut(data=item_choice):
    return_list = list()
    for i, d in enumerate(data):
        cut_data = jieba.cut(d)
        return_list.append(" ".join(cut_data))
        if i % 20000 == 0:
            print('Now is steps %d '%i)
    return return_list

# define the get key_words method
def get_keywords(data=item_choice,topk=4, iter=2):
    start_time = time.time()
    key_words = data
    key_words_return = list()
    for j in range(iter):
        print('Now in the iteration %d'%j)
        for i,d in enumerate(key_words):
            key_words_return.append(ana.extract_tags(d,topK=topk,allowPOS=('n','nr')))
            if(i%20000 == 0):
                print('now is the %d samples for iteration %d'%(i, j))
        key_words = list()
        for m in range(len(key_words_return)):
            key_words.append("".join(key_words_return[m]))
        if(j != iter - 1):
            key_words_return = list()
    print('The all process used %f seconds'%(time.time()-start_time))
    return key_words_return
# Get the key_words list
topk = 6
key_words = get_keywords(topk=topk, iter=1)

#make the key-words using word2vec
def get_word2vec(sentences,min_count=1,iter=500,size=100,workers=5,window=3, sg=1):
    start_time = time.time()
    model = Word2Vec(sentences,min_count=min_count,iter=iter,size=size,workers=workers,max_vocab_size=None,window=window,sg=sg)
    model_path = 'F:\workingData\\201806\\recommendation\MultiLabel\multi_model'
    # persist the trained model to disk
    model.save(model_path+'/word2vec.bin' ,protocol=2)
    # load the trained model
    model = Word2Vec.load(model_path+'/word2vec.bin')
    wordsvec = model[model.wv.vocab]
    uni_words = list(model.wv.vocab)
    print('The word2vec model takes %f seconds'%(time.time()-start_time))
    return wordsvec,uni_words
#get the word2vec result and the unique words list
# original iter is 30 for residual network-8 get 0.886 accuracy
wordsvec, uni_words = get_word2vec(cut_words_with_attr_list,window=5, iter=100,min_count=5)


""" using the already trained model for the word using the fastText """
def trained_model():
    s_t = time.time()
    trained_path = 'E:\github\\fastText\chineseModels\cc.zh.300.vec'
    from gensim.models import KeyedVectors
    new_word_key = KeyedVectors.load_word2vec_format(trained_path)
    wordsvec = new_word_key[new_word_key.wv.vocab]
    uni_words = new_word_key.wv.vocab
    print('total load time %.2f seconds'%(time.time()-s_t))
    return wordsvec, uni_words
#  wordsvec, uni_words = trained_model()

# make the key-words and word2vec result directory for bellow using the key-value
res_dic = dict()
for j in range(len(uni_words)):
    res_dic.update({uni_words[j]: wordsvec[j, :]})

# make the label columns to be list type. Used in case for the key_word list is null, so just remove the indexed label
label_list = np.array(label_choice)
key_words_list = np.array(key_words)

# get the null row nums
null_col = np.empty_like(key_words_list)
for i in range(len(key_words)):
    if (len(key_words[i]) == 0):
        null_col[i] = False
    else:
        null_col[i] = True
# make the data type to be boolean
null_col = null_col.astype(np.bool)
# get the non-null cols
label_training = label_list[null_col]
key_words_training = key_words_list[null_col]

# test the sklearn tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
# tm = TfidfVectorizer().fit(key_words_training).transform(key_words_training)
key_list = key_words_training.tolist()

tm = list()
for i in range(len(key_list)):
    tm.append(" ".join(key_list[i]))
vec_model = TfidfVectorizer(min_df=1)
d = vec_model.fit_transform(tm)
idf = vec_model.idf_
r_idf = dict(zip(vec_model.get_feature_names(),idf))

# convert the list_in_array to be a just array
# compute the tf-idf array for next step to multiply with the each word vector
conver_array = np.empty_like(key_words_training)
for i in range(len(key_words_training)):
    tfidf_list = list()
    for j in range(len(key_words_training[i])):
        # first compute the tf value
        tf = key_words_training[i].count(key_words_training[i][j])/len(key_words_training[i])
        if(key_words_training[i][j] not in r_idf.keys()):
            #idf = 1.   # change the not exist key-word for 0.0 means that the key-word shows in all corpus log2(n/n) =0.0
            idf = 0.0
        else: idf = r_idf[key_words_training[i][j]]
        tf_idf = tf*idf
        tfidf_list.extend([tf_idf])
    conver_array[i] = tfidf_list

#make the key_words sentences vecter by meaning the sum of the all vectors number
# note we need to substract the null_row_num
result = np.zeros((key_words_training.shape[0],wordsvec.shape[1]))
# convert unique list to set is much faster
res_keys = set(list(res_dic.keys()))
# loop the all visuable vectors
for i in range(result.shape[0]):
    for j in range(len(key_words_training[i])):
        # because of the now word2vec model is based on the all satisified word, need to get the satisified key-word vector
        if(key_words_training[i][j] not in res_keys):
            continue
        # add the tf-idf vector to the res
        result[i,:] = res_dic[key_words_training[i][j]] * conver_array[i][j]
    result[i,:] = result[i,:]/len(key_words_training[i])

# result dimension like num*100*5 tensor
result_image = np.zeros((key_words_training.shape[0], topk, wordsvec.shape[1]))
# convert unique list to set is much faster
res_keys = set(list(res_dic.keys()))
# loop the all visuable vectors
for i in range(result.shape[0]):
    for j in range(len(key_words_training[i])):
        # because of the now word2vec model is based on the all satisified word, need to get the satisified key-word vector
        if(key_words_training[i][j] not in res_keys):
            continue
        # add the tf-idf vector to the res
        result_image[i,j,:] = res_dic[key_words_training[i][j]] * conver_array[i][j]

# get the training data and label
train_data = result
label = np.array(label_training).astype(np.int32)

# split the data to train and test data
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(train_data, label, test_size=.2, random_state=1234)
num_classes = np.unique(label).shape[0]

# split the image-like data to be train and test datasets
label_l = label.reshape(-1,1)
label_o = keras.utils.to_categorical(label_l, num_classes=num_classes)
num_classes = np.unique(label_l).shape[0]
xtrain_image, xtest_image, ytrain_image, ytest_image = train_test_split(result_image, label_o, test_size=.15, random_state=1234)
print(xtrain_image.shape)


# for the lstm model to split data to train data and test data
xtrain_l, xtest_l, ytrain_l, ytest_l = train_test_split(train_data, label_o, test_size=.2, random_state=1234)
# reshape the data to be n*10*10
data_lstm = train_data.reshape(-1,10,10)
label_lstm = label.reshape(-1,1)

# make the label to be one-hot encoding
label_lstm_o = keras.utils.to_categorical(label_lstm, num_classes=np.unique(label_lstm).shape[0])

# split the lstm data to be train and test data
xtrain_lstm, xtest_lstm, ytrain_lstm, ytest_lstm = train_test_split(data_lstm, label_lstm_o, test_size=.1, random_state=1234)




""" define a leaky_relu function for activation function """
def leaky_relu(x, alpha=.2):
    return .5*(1+alpha) + .5*(1-alpha)*np.abs(x)

# following is all the deep model structures
# including CNN-1D, CNN-2D, LSTM, Stacked-LSTM, Residual, Wide-residual, Inception
def CNN_model_con(compile=True, op=None, dropout=.5):
    model = Sequential()

    model.add(Conv1D(256, 2, activation='relu', input_shape=(7, 100)))
    model.add(Conv1D(256, 2, activation='relu'))
    model.add(Dropout(dropout))
    # model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Conv1D(256, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))

    if compile:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    else:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=op)

    print('model structure')
    model.summary()
    return model


# build a residual network for each layer will be added the input data
# using the residual network added with batch normalization
# I have used the batch_normalization to train the model, I find that
# it is almost the same accuracy curve compared with the original network
# which was inspired by DenseNet(which block layer will just be added with the input)
def res_input_model(num_layers=8, fc_needed=True, fc_units=512, needed_dropout=True,
                    dropout=.5, data=xtrain_image, batchnorm=True,
                    is_global=True, is_flatten=False, act_fun='relu'):
    num_rows, num_cols = data.shape[1], data.shape[2]
    inputs = Input(shape=(num_rows, num_cols))

    # wrap the residual block into function
    def _res_block(layer_out, is_first=False):
        layer = Conv1D(num_cols, 2, padding='same')(layer_out)
        if batchnorm:
            layer = BatchNormalization()(layer)
        # just change the activation function to be layer
        if act_fun == 'relu':
            layer = Activation(act_fun)(layer)
        else:
            # for a user defined activation function
            get_custom_objects().update({'user_activation':Activation(act_fun)})
            layer = Activation(act_fun)(layer)
        if needed_dropout:
            layer = Dropout(dropout)(layer)
        if is_first:
            return layer
        else: return keras.layers.add([inputs, layer])

    layer = _res_block(inputs, is_first=True)
    for i in range(num_layers - 1):
        layer = _res_block(layer)
    if is_global:
        layer = GlobalAveragePooling1D()(layer)
    if is_flatten:
        layer = Flatten()(layer)

    if (fc_needed):
        layer = Dense(fc_units)(layer)
        layer = BatchNormalization()(layer)
        if act_fun == 'relu':
            layer = Activation('relu')(layer)
        else:
            # for a user defined activation function
            get_custom_objects().update({'user_activation':Activation(act_fun)})
            layer = Activation(act_fun)(layer)
        layer = Dropout(.5)(layer)

    pred = Dense(num_classes, activation='softmax')(layer)

    model = Model(inputs=[inputs], outputs=pred)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())

    print('Model Struture:')
    model.summary()

    return model


# using the residual network blocks
# This one is the deep residual network for each output will be added to next layer
# build a basic residual network that the next layers' input is getted from the second
# before layers' inputs added with the current output
def res_basic_model(num_layers=4, fc_needed=True, fc1_units=512, dropout=.5, data=xtrain_image,
                    is_dropout=True, is_batchnorm=True):
    data_3_dimension = data.shape[2]
    inputs = Input(shape=(data.shape[1], data_3_dimension))

    def _res_block(layer_out=inputs):
        layer1 = Conv1D(data_3_dimension, 2, padding='same')(layer_out)
        if is_batchnorm:
            layer1 = BatchNormalization()(layer1)
        layer1 = Activation('relu')(layer1)
        if is_dropout:
            layer1 = Dropout(dropout)(layer1)

        layer2 = Conv1D(data_3_dimension, 2, padding='same')(layer1)
        if is_batchnorm:
            layer2 = BatchNormalization()(layer2)
        layer2 = Activation('relu')(layer2)
        if is_dropout:
            layer2 = Dropout(dropout)(layer2)
        return keras.layers.add([layer1, layer2])

    layer = _res_block()
    num_layers = num_layers
    for i in range(num_layers - 1):
        layer = _res_block(layer_out=layer)

    global_out = GlobalAveragePooling1D()(layer)

    if(fc_needed):
        global_out = Dense(fc1_units)(global_out)
        if is_batchnorm:
            global_out = BatchNormalization()(global_out)
        global_out = Activation('relu')(global_out)
        if is_dropout:
            global_out = Dropout(dropout)(global_out)

    pred = Dense(num_classes, activation='softmax')(global_out)

    model = Model(inputs=[inputs], outputs=pred)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())

    print('Model Struture:')
    model.summary()

    return model


# build a wide-deep model using residual networks like-struture
# using the wide residual network papers to build the wide_residual network
# for now is just a 4-inceptions for residual, add the input data for each residual layer output
def wide_deep_model(num_layers=3, added_inputs=False, fc_needed=True,
                    fc_units=512, conv_needed=True, conv_units=256,
                    is_global=True):
    inputs = Input(shape=(xtrain_image.shape[1], xtrain_image.shape[2]))

    def __add_batch_norm(layer):
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(.5)(layer)
        return layer

    def _wide_deep_block(inputs=inputs):
        wide1_1 = Conv1D(100, 2, padding='same')(inputs)
        wide1_1 = __add_batch_norm(wide1_1)
        wide1_2 = Conv1D(100, 1, padding='same')(inputs)
        wide1_2 = __add_batch_norm(wide1_2)
        wide1_2 = Conv1D(100, 2, padding='same')(wide1_2)
        wide1_2 = __add_batch_norm(wide1_2)
        # wide1_3 = Conv1D(100, 1, padding='same')(inputs)
        # wide1_3 = __add_batch_norm(wide1_3)
        # wide1_3 = Conv1D(100, 2, padding='same')(wide1_3)
        # wide1_3 = __add_batch_norm(wide1_3)
        wide1_4 = Conv1D(100, 1, padding='same')(inputs)
        wide1_4 = __add_batch_norm(wide1_4)

        if (added_inputs):
            return keras.layers.add([inputs, wide1_1, wide1_2, wide1_4])
        else:
            return keras.layers.add([wide1_1, wide1_2, wide1_4])

    wide_layer = _wide_deep_block()
    for _ in range(num_layers):
        wide_layer = _wide_deep_block(wide_layer)

    if (conv_needed):
        # add new convolutional networks
        wide_layer = Conv1D(conv_units, 2)(wide_layer)
        wide_layer = __add_batch_norm(wide_layer)
        wide_layer = Conv1D(conv_units, 2)(wide_layer)
        wide_layer = __add_batch_norm(wide_layer)

    # using the global averaging
    if is_global:
        wide_layer = GlobalAveragePooling1D()(wide_layer)

    if (fc_needed):
        # adding a dense layer
        wide_layer = Dense(fc_units)(wide_layer)
        wide_layer = BatchNormalization()(wide_layer)
        wide_layer = Activation('relu')(wide_layer)
        wide_layer = Dropout(.5)(wide_layer)
    pred = Dense(num_classes, activation='softmax')(wide_layer)

    # construct the model and compile the model
    model = Model(inputs=[inputs], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    print('Model Structure:')
    model.summary()

    return model



# inspired by the Fully Convolutional network structure, build a fully convolution nets
def cnn_1d_fcn():
    model = Sequential()
    model.add(Conv1D(512, 1, activation='relu', input_shape=(topk, 100), padding='same'))
    model.add(Dropout(.5))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Dropout(.5))
    model.add(Conv1D(512, 1, activation='relu'))
    model.add(Dropout(.5))
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Dropout(.5))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    print('Model structure')
    model.summary()
    return model

def cnn_lstm_combine():
    model = Sequential()
    model.add(Conv1D(512, 1, activation='relu', input_shape=(topk, 100), padding='same'))
    model.add(Conv1D(256, 1, activation='relu', padding='same'))
    model.add(Conv1D(256, 1, activation='relu', padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    print('Model Structure:')
    model.summary()
    return model


def lstm_model():
    model = Sequential()
    model.add(Embedding(100, output_dim=256))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.5))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    print('Model Structure')
    model.summary()
    return model


def stacked_lstm_model(num_layers=2):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(topk, 100)))
    model.add(Dropout(.5))
    for _ in range(num_layers):
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(.5))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    print('Model Structure')
    model.summary()
    return model


def stacked_lstm_model_stronger():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(topk, 100)))
    model.add(Dropout(.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.5))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(.5))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(.5))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())
    print('Model Structure')
    model.summary()
    return model


def plot_acc(history, title=None):
    acc_list = history.history['acc']
    loss_list = history.history['loss']
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].set_title(title)
    ax[0].plot(acc_list, label='adagrad')
    ax[0].set_xticks(np.arange(len(acc_list)))
    ax[0].set_title('accuracy')
    ax[1].set_title(title)
    ax[1].plot(loss_list, label='adagrad')
    ax[1].set_xticks(np.arange(len(loss_list)))
    ax[1].set_title('loss')
    plt.legend()
    plt.show()


def plot_acc_dict(hist_dict, op_list, title='Different Optimizer'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plt.title(title)
    for i in range(len(hist_dict)):
        acc_list = hist_dict[op_list[i]].history['acc']
        loss_list = hist_dict[op_list[i]].history['loss']
        ax[0].plot(acc_list, label=op_list[i])
        ax[0].set_title('accuracy')
        ax[0].set_xticks(np.arange(len(acc_list)))
        # plt.legend()
        ax[1].plot(loss_list, label=op_list[i])
        ax[1].set_title('loss')
        ax[1].set_xticks(np.arange(len(loss_list)))
    plt.legend()
    plt.show()


# Different batch_size affects the accuracy
def diff_batch_cnn():
    batch_dict = dict()
    batch_list = [512, 1024, 2048]
    for batch in batch_list:
        model = CNN_model_con()
        batch_dict[batch] = model.fit(xtrain_image, ytrain_image, batch_size=batch, epochs=10)
        # plot_acc_for_cnn(history, title=batch_list)
    plot_acc_dict(batch_dict)


# different optimizer affects the accuracy
# during training, I find the adadelta is much better than sgd
def diff_opimizer_cnn():
    hist_dict_cnn = dict()
    acc_test_list = list()
    sgd = keras.optimizers.SGD(lr=.0001, momentum=.9, decay=1e-5, nesterov=True)
    rmsprom = keras.optimizers.rmsprop(lr=0.0001)
    adam = keras.optimizers.adam(lr=.0001)
    adagrad = keras.optimizers.adagrad()
    adadelta = keras.optimizers.adadelta()
    adamax = keras.optimizers.adamax(lr=.0001)
    op_list = ['sgd', 'adam', 'Adagrad']
    op_dict = {op_list[0]: sgd, op_list[1]: adam,
               op_list[2]: adagrad}
    # loop the op_dict
    for i in range(len(op_list)):
        model = CNN_model_con(compile=False, op=op_dict[op_list[i]])
        hist_dict_cnn[op_list[i]] = model.fit(xtrain_image, ytrain_image, batch_size=512, epochs=10)
        acc_test_list.append(model.evaluate(xtest_image, ytest_image, batch_size=512))
    plot_acc_dict(hist_dict_cnn)
    plt.plot(acc_test_list)
    plt.show()


# wrap the train and evaluate method
def train_model(model, epochs=20, batch_size=1024):
    s_t = time.time()
    history = model.fit(xtrain_image, ytrain_image, batch_size=batch_size, epochs=epochs)
    e_t = time.time()
    print('Model Training use %.2f seconds'%(e_t - s_t))
    return history

def eval_model(model):
    acc = model.evaluate(xtest_image, ytest_image, batch_size=2048)[1]
    print('Model Accuracy = %.4f'%(acc))
    return acc


# wrap a function to train the different deep residual network
def residual_compare():
    depths = [16, 32, 64, 128, 256]
    list_model = ['res_16','res_32','res_64','res_128', 'res_256']
    hist_dict = dict()
    acc_list = list()
    for i in range(len(depths)):
        model = res_input_model(num_layers=depths[i], fc_needed=True, batch_size=2048)
        history = train_model(model)
        hist_dict[list_model[i]] = history
        acc_list.append(eval_model(model))
    # plot the different structure accuracy curve
    #plot_acc_dict(hist_dict, list_model)
    print('Model accuracy list :',acc_list)
    return acc_list


# wrap the wide_deep model to train the model
def wide_deep_compare():
    depths = [5, 7, 9]
    list_model = ['wide_deep_5', 'wide_deep_7', 'wide_deep_9']
    hist_dict = dict()
    acc_list = list()
    for i in range(len(depths)):
        model = wide_deep_model(num_layers=depths[i],added_inputs=True)
        history = train_model(model)
        hist_dict[list_model[i]] = history
        acc_list.append(eval_model(model))
    print('Model Accuracy list :',acc_list)
    return hist_dict, acc_list

# wrap stacked LSTM model
def stack_compare():
    layers = [3, 5, 7, 9]
    list_model = ['stacked_3', 'stacked_5', 'stacked_7', 'stacked_9']
    hist_dict = dict()
    acc_list = list()
    for i in range(len(layers)):
        model = stacked_lstm_model(num_layers=layers[i])
        history = train_model(model)
        hist_dict[list_model[i]] = history
        acc_list.append(eval_model(model))
    print('Model Accuracy List : ', acc_list)
    return acc_list

#diff_wide_dict, diff_wide_acc_list = wide_deep_compare()
# diff_acc = residual_compare()
diff_list = stack_compare()




# using the tensorflow-hub to do the vector assign for transfer learning
import tensorflow as tf
import tensorflow_hub as hub
from itertools import chain

""" get the Chinese neural network language models for 128 with normalization """
embed = hub.Module("https://tfhub.dev/google/nnlm-zh-dim128-with-normalization/1")
# get all the cut_words set
cut_data_list = jieba_cut()
cut_words_sets = list(set(m for m in chain(*cut_data_list)))
embeddings_org_data = embed(cut_words_sets)
# embeddings = embed(cut_words_sets)
# get the pretrained vectors for the tensorflow-hub
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    result_image_tensor = sess.run(embeddings_org_data)
# get the directory of the key-vector
res_dic_tensor = dict()
for i in range(result_image_tensor.shape[0]):
    res_dic_tensor[cut_words_sets[i]] = result_image_tensor[i, :]

# get the output tensor result
result_image_tensor_out = np.zeros((key_words_training.shape[0], topk, result_image_tensor.shape[1]))
# convert unique list to set is much faster
res_keys_tensor = set(list(res_dic_tensor.keys()))
# loop the all visuable vectors
for i in range(result_image_tensor.shape[0]):
    for j in range(len(key_words_training[i])):
        # because of the now word2vec model is based on the all satisified word, need to get the satisified key-word vector
        if(key_words_training[i][j] not in res_keys_tensor):
            continue
        # add the tf-idf vector to the res
            result_image_tensor_out[i,j,:] = res_dic_tensor[key_words_training[i][j]] * conver_array[i][j]

""" split the data to be train and test data sets for the tensorflow-hub """
xtrain_tensor, xtest_tensor, ytrain_tensor, ytest_tensor = train_test_split(result_image_tensor_out, label_o, test_size=.2, random_state=1234)
def train_model_tensor(model, epochs=20, batch_size=1024):
    s_t = time.time()
    history = model.fit(xtrain_tensor, ytrain_tensor, batch_size=batch_size, epochs=epochs)
    e_t = time.time()
    print('Model Training use %.2f seconds'%(e_t - s_t))
    return history

def eval_model_tensor(model):
    acc = model.evaluate(xtest_tensor, ytest_tensor, batch_size=1024)[1]
    print('Model Accuracy = %.4f'%(acc))
    return acc



""" using the original datasets to map the datasets to 128-D matrix"""
# just using the items dataframe for model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    result_org_data = sess.run(embeddings_org_data)
# split the result data and label to train and test
label_o_tensor = keras.utils.to_categorical(np.array(new_label).reshape(-1,1), num_classes=num_classes)
xtrain_tensor_org, xtest_tensor_org, ytrain_tensor_org, ytest_tensor_org = train_test_split(result_org_data, np.array(new_label), test_size=.2, random_state=1234)
def train_model_org_tensor(model, epochs=20, batch_size=1024):
    s_t = time.time()
    history = model.fit(xtrain_tensor_org, ytrain_tensor_org, batch_size=batch_size, epochs=epochs)
    e_t = time.time()
    print('Model Training use %.2f seconds'%(e_t - s_t))
    return history

def eval_model_org_tensor(model):
    acc = model.evaluate(xtest_tensor_org, ytest_tensor_org, batch_size=1024)[1]
    print('Model Accuracy = %.4f'%(acc))
    return acc

""" for TensorFlow high level API, build a input_fn for train and test """
def input_fn(data, label, epochs=5000, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(x={'x':data}, y=label, num_epochs=epochs, shuffle=shuffle)
feature_columns = tf.feature_column.numeric_column('x',shape=(result_org_data.shape[1],))

train_input_fn = input_fn(xtrain_tensor_org, ytrain_tensor_org)
test_input_fn = input_fn(xtest_tensor_org, ytest_tensor_org, epochs=1, shuffle=False)
model_path = 'F:\workingData\\201804\\taobao\models\\tensorboard'
# linear model
estimator = tf.estimator.LinearClassifier(n_classes=num_classes,
                                          optimizer=tf.train.FtrlOptimizer(learning_rate=.1, l2_regularization_strength=1.),
                                          feature_columns=[feature_columns],
                                          model_dir=model_path)
# dnn model
estimator_dnn = tf.estimator.DNNClassifier(n_classes=num_classes, optimizer=tf.train.AdadeltaOptimizer(),
                                           feature_columns=[feature_columns], hidden_units=[256, 512, 256],
                                           dropout=.5,
                                           model_dir=model_path)



""" using the auto-encoder to reconstruct the data """
def auto_encoder_conv():
    x = Input(shape=(xtrain_image.shape[1], xtrain_image.shape[2]))

    # encoder part
    encoder_conv1 = Conv1D(100, 2, activation='relu', padding='same')(x)
    encoder_pool1 = MaxPooling1D(2)(encoder_conv1)
    encoder_conv2 = Conv1D(100, 2, activation='relu', padding='same')(encoder_pool1)
    #encoder_out = GlobalAveragePooling1D()(encoder_conv2)

    # decoder part
    decoder_conv1 = Conv1D(100, 2, activation='relu', padding='same')(encoder_conv2)
    decoder_uppool1 = UpSampling1D(2)(decoder_conv1)
    decoder_out = Conv1D(100, 2, activation='relu', padding='same')(decoder_uppool1)

    auto_encoder = Model(inputs=x, outputs=decoder_out)
    auto_encoder.compile(optimizer='adam', loss='mse')

    print('Model Structure:')
    auto_encoder.summary()

    return auto_encoder

def auto_encoder_mul(data=xtrain_image):
    data = data.reshape(-1,7*100)
    in_size = data.shape[1]
    hidden_size = 256
    code_size = 64
    x = Input(shape=(in_size,))

    encoder_1 = Dense(hidden_size, activation='relu')(x)
    h = Dense(code_size, activation='relu')(encoder_1)

    decoder_1 = Dense(hidden_size, activation='relu')(h)
    out = Dense(in_size, activation='relu')(decoder_1)

    model = Model(input=x, output=out)
    model.compile(loss='mse', optimizer='adam')
    print('Model Structures:')
    model.summary()
    return model
