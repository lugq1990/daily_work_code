# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import logging
import pymysql
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
import time

sc = SparkContext()
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
# connection = pymysql.connect(user='zhanghui',password='zhanghui',database='model_data',host='10.1.36.18',charset='utf8')

# query = "select a.*,b.class_score from model_data.train_data_X_cnn a left join  model_data.train_data_y_cnn b on a.mobile = b.mobile"
#
# # just for test how much time for reading mysql
# def read_data():
#     print('Start read mysql database:')
#     s_t = time.time()
#     data_sql = pd.read_sql(query, con=connection)
#     e_t = time.time()
#     print('Total used %.2f seconds'%(e_t - s_t))
#     return data_sql
# data_sql= read_data()
# data_sql.drop(['day_no','maybe_shopping'], axis=1, inplace=True)

path = '/home/lugq/data/recommend'
# local_path = 'F:\workingData\\201806\ALS\\train_data'
local_path = 'F:\workingData\\201806\\recommendation\\train_data'
# df = pd.read_csv(local_path+'/train_data_cnn.csv')
# df = pd.read_csv(local_path+'/train_data_cnn_11.csv')
df_public = pd.read_csv(local_path+'/train_data_clothing_lstm_12.csv')


# get the data and label column data
data = df_public.iloc[:,1:-1]
label = df_public.iloc[:,-1]

# data = np.array(data).reshape(-1, 53, 12)
data = np.array(data).reshape(-1, 12, 12)
label = np.array(label).reshape(-1, 1)

# split the data and label to train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

# model = Sequential()
#
# model.add(Conv1D(256, 2, activation='relu', input_shape=(data.shape[1], data.shape[2])))
# model.add(Conv1D(256, 2, activation='relu'))
# model.add(Dropout(.5))
# # model.add(MaxPooling1D(2))
# model.add(Conv1D(256, 2, activation='relu'))
# model.add(Dropout(.5))
# model.add(Conv1D(256, 2, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(1))
#
# model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adadelta())
# print('Model structure:')
# model.summary()

""" Use Tensorflow to build the Convolutional Nets"""
import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(256, 2, activation='relu', input_shape=(data.shape[1], data.shape[2])))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Conv1D(256, 2, activation='relu', padding='same'))
model.add(tf.keras.layers.Dropout(.5))
# model.add(tf.keras.layers.Conv1D(256, 2, activation='relu', padding='same'))
# model.add(tf.keras.layers.Dropout(.5))
# model.add(tf.keras.layers.Conv1D(256, 2, activation='relu', padding='same'))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(lr=1e-5))
model.summary()




""" Because of ipython can not get the function, so manually build the model"""
import keras
from keras.models import Model
from keras.layers import BatchNormalization,Activation,Dense,Dropout,Input,Flatten
from keras.utils.generic_utils import get_custom_objects

def res_input_model(num_layers=8, fc_needed=True, fc_units=512, needed_dropout=True,
                    dropout=.5, data=data, batchnorm=True,
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

    pred = Dense(51)(layer)

    model = Model(inputs=[inputs], outputs=pred)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())

    print('Model Struture:')
    model.summary()

    return model


"""build a LSTM model, this is the best model for SME loss:20 """
""" we can simply modify the final layer output to make the multi-class problem"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adadelta())
model.summary()

model.fit(xtrain, ytrain, epochs=100, batch_size=512)

# combine different batch_size for test loss and accuracy
batch_size = [128, 512, 1024]
def combine_diff_batch(batch_list=batch_size):
    loss_list = list()
    for i in range(len(batch_list)):
        history_lstm = model.fit(xtrain, ytrain, epochs=100, batch_size=batch_list[i])
        loss_list.append(history_lstm.history['loss'])
    return loss_list

""" get the satisfied columns"""

def get_data(index=4):
    res_data = df_public.iloc[:, index]
    for i in range(12 - 1):
        res_data = pd.concat((res_data, df_public.iloc[:, index + 12*i]), axis=1)
    return res_data




""" Bellow use the mult-ilable algorithm to train the model"""
new_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
df = pd.read_csv(new_path+'/train_data_12.csv')
df.dropna(inplace=True)
data = df.iloc[:,:145].drop('mobile', axis=1)
label = df.iloc[:, 145:]

# spilt the data to train and test
xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
classifier_basic = OneVsRestClassifier(lr)
# start to train the model
classifier_basic.fit(xtrain, ytrain)
pred = classifier_basic.predict(xtest)


# for the multi-label problem, define a function for different algorithms accuracy plot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import time
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
import matplotlib.pyplot as plt

dtc = DecisionTreeClassifier(random_state=1234)
rfc = RandomForestClassifier(random_state=1234)
gbc = GradientBoostingClassifier(random_state=1234)
clf = SVC(C=10, random_state=1234)
abc = AdaBoostClassifier(random_state=1234)
algorithm_dict = {'dtc':dtc, 'rfc':rfc,'abc':abc, 'svm':clf}
strategy_dict = {'Chain': ClassifierChain}

def multi_label_com(algorithm_dict=algorithm_dict, strategy_dict=strategy_dict):
    s_t = time.time()
    res_dic = dict()
    alg_keys = list(algorithm_dict.keys())
    str_keys = list(strategy_dict.keys())
    for j in range(len(strategy_dict)):
        for i in range(len(algorithm_dict)):
            classifier = strategy_dict[str_keys[j]](algorithm_dict[alg_keys[i]])
            classifier.fit(xtrain, ytrain)
            pred = classifier.predict(xtest)
            acc = metrics.accuracy_score(ytest, pred)
            res_dic[alg_keys[i] + str_keys[j]] = acc
    e_t = time.time()
    print('Total training time %.2f seconds'%(e_t - s_t))
    plt.scatter(res_dic.keys(), res_dic.values())
    plt.show()
    return res_dic


""" we can simply modify the final layer output to make the multi-class problem"""
if data.shape[1] == 144:
    new_data = np.array(data).reshape(-1, 12, 12)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(new_data.shape[1], new_data.shape[2])))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(label.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta())
model.summary()

xtrain, xtest, ytrain, ytest = train_test_split(new_data, label, test_size=.2, random_state=1234)


""" Now build a auto-encoder model to get new features from original datasets,
    Bellow code use TensorFlow to build an auto-encoder
"""
auto_data = np.array(data)
# first define all the hyperparameters
learning_rate = .001
steps = 1000
display_step = 5
tensor_path = 'F:\workingData\\201806\\recommendation\MultiLabel\\tensorboard'
# this is the all size of data and layers numbers
n_input = data.shape[1]
n_hidden_1 = 256
n_hidden_2 = 256
# then define the input placeholders for inputs and targets
x = tf.placeholder(tf.float32, [None, n_input], name='inputs')
y = tf.placeholder(tf.float32, [None, n_input], name='targets')

# init all the weights and biases using tf.variables, using Gaussian random
weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='encoder_h1'),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='encoder_h2'),
           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name='decoder_h1'),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name='decoder_h2')}
biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name='encoder_b1'),
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]),name='encoder_b2'),
          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name='decoder_b1'),
          'decoder_b2': tf.Variable(tf.random_normal([n_input]), name='decocer_b2')}
# now is encoder layers
def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['encoder_h2']), biases['encoder_b2']))
    return layer2
# this is decoder layer
def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['decoder_h2']), biases['decoder_b2']))
    return layer2
# now we can build the models
with tf.name_scope('encoder_op'):
    encoder_op = encoder(x)
with tf.name_scope('decoder_op'):
    decoder_op = decoder(encoder_op)

# we can get the prediction of the decoder output as model prediction
with tf.name_scope('prediction'):
    pred = decoder_op

# now is the loss, use the MSE
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.pow(y - pred, 2))
# this is the optimizer to optimize the loss
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(loss)
# after all the optimizer have been defined, so we can init all the variables of the model
init = tf.global_variables_initializer()

# before start to train the model, we can summary the loss, we also have to get the all merge-summary op
tf.summary.scalar('loss', loss)
merge_op = tf.summary.merge_all()

# start to train the model
sess = tf.Session()
sess.run(init)
loss_list = list()
# for TensorBoard using, we have to make a filewriter
# summary_writer = tf.summary.FileWriter(tensor_path, graph=tf.get_default_graph())
s_t = time.time()
for i in range(steps):
    _, l_s = sess.run([optimizer, loss],
                              feed_dict={x:auto_data, y:auto_data})
    loss_list.append(l_s)
    if i % display_step == 0:
        print('Now is step %d, loss = %.6f'%(i, l_s))
        # summary_writer.add_summary(summary)
e_t = time.time()
print('Total training cost %.2f seconds'%(e_t - s_t))
# start to plot the loss fun
plt.plot(loss_list)
plt.title('Loss curve')
plt.show()





"""
I just want to research how much the hidden units of the auto-encoder model will be best for the model
"""
def auto_encoder_res(hidden_units = 256,steps = 400, display_step=20):
    auto_data = np.array(data)
    # first define all the hyperparameters
    learning_rate = .001
    steps = steps
    display_step = display_step
    tensor_path = 'F:\workingData\\201806\\recommendation\MultiLabel\\tensorboard'
    # this is the all size of data and layers numbers
    n_input = data.shape[1]
    n_hidden_1 = hidden_units
    n_hidden_2 = hidden_units
    # then define the input placeholders for inputs and targets
    x = tf.placeholder(tf.float32, [None, n_input], name='inputs')
    y = tf.placeholder(tf.float32, [None, n_input], name='targets')

    # init all the weights and biases using tf.variables, using Gaussian random
    weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='encoder_h1'),
               'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='encoder_h2'),
               'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name='decoder_h1'),
               'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name='decoder_h2')}
    biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name='encoder_b1'),
              'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]), name='encoder_b2'),
              'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name='decoder_b1'),
              'decoder_b2': tf.Variable(tf.random_normal([n_input]), name='decocer_b2')}

    # now is encoder layers
    def encoder(x):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['encoder_h2']), biases['encoder_b2']))
        return layer2

    # this is decoder layer
    def decoder(x):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['decoder_h2']), biases['decoder_b2']))
        return layer2

    # now we can build the models
    with tf.name_scope('encoder_op'):
        encoder_op = encoder(x)
    with tf.name_scope('decoder_op'):
        decoder_op = decoder(encoder_op)

    # we can get the prediction of the decoder output as model prediction
    with tf.name_scope('prediction'):
        pred = decoder_op

    # now is the loss, use the MSE
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(y - pred, 2))
    # this is the optimizer to optimize the loss
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    # after all the optimizer have been defined, so we can init all the variables of the model
    init = tf.global_variables_initializer()

    # before start to train the model, we can summary the loss, we also have to get the all merge-summary op
    tf.summary.scalar('loss', loss)
    merge_op = tf.summary.merge_all()

    # start to train the model
    sess = tf.Session()
    sess.run(init)
    loss_list = list()
    # for TensorBoard using, we have to make a filewriter
    # summary_writer = tf.summary.FileWriter(tensor_path, graph=tf.get_default_graph())
    s_t = time.time()
    for i in range(steps):
        _, l_s = sess.run([optimizer, loss],
                          feed_dict={x: auto_data, y: auto_data})
        loss_list.append(l_s)
        if i % display_step == 0:
            print('Now is step %d, loss = %.6f' % (i, l_s))
            # summary_writer.add_summary(summary)
    e_t = time.time()
    print('Total training cost %.2f seconds' % (e_t - s_t))

    res = sess.run(encoder_op, feed_dict={x:auto_data})
    return res

# then I will get the auto-encoder result and use LSTM to train the new generated data
hidden_list = [256, 400, 625]  # means the model input size is 16*16, 20*20, 15*25
# also abstract the model training process to be a function
def compare_diff_hiddens(hidden_list=hidden_list, epochs=100):
    acc_dict = dict()
    eval_acc = dict()
    for hiddens in hidden_list:
        train_data = auto_encoder_res(hidden_units=hiddens).reshape(-1, np.sqrt(hiddens).astype(np.int32),
                                                                    np.sqrt(hiddens).astype(np.int32))
        # we just use all the data to train the lstm model, but we also have to recontruct the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(train_data.shape[1], train_data.shape[2])))
        model.add(tf.keras.layers.Dropout(.5))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True))
        model.add(tf.keras.layers.Dropout(.5))
        model.add(tf.keras.layers.LSTM(128))
        model.add(tf.keras.layers.Dense(label.shape[1], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta())
        model.summary()
        # start to train the model, split the dataset to train and test
        xtrain_t, xtest_t, ytrain_t, ytest_t = train_test_split(train_data, label, test_size=.2, random_state=1234)
        his = model.fit(xtrain_t, ytrain_t, epochs=epochs, batch_size=512)
        acc_dict[hiddens] = his
        eval_acc[hiddens] = model.evaluate(xtest_t, ytest_t, batch_size=1024)

    his_res = list(acc_dict.values())
    his_keys = list(acc_dict.keys())
    # plot different accuracy result in one plot
    fig, ax = plt.subplots(2, 1)
    for m in range(len(his_res)):
        ax[0].plot(his_res[m].history['acc'], label='hidden_units_'+np.str(his_keys[m]))
        ax[0].set_title('Different hidden units accuracy')
        ax[1].plot(his_res[m].history['loss'], label='hidden_units_'+np.str(his_keys[m]))
        ax[1].set_title('Different hidden units loss')
    plt.legend()
    plt.show()
    return acc_dict, eval_acc
acc_dict, eval_acc = compare_diff_hiddens()


"""
 This bellow just use the deep model as feature extracting algorithm, and use the
 extracted features to feed to LSTM or CNN or some basic machine learning algorithms,
 this model is just trained on the original regression problem LSTM model,
 train it, and get the model's final layer's outputs as the new datasets
"""
path = 'F:\workingData\\201806\ALS\\train_data'
# local_path = 'F:\workingData\\201806\\recommendation\\train_data'
# df = pd.read_csv(local_path+'/train_data_cnn.csv')
df = pd.read_csv(path+'/train_data_cnn_11.csv')
# df_public = pd.read_csv(local_path+'/train_data_clothing_lstm_12.csv')


# get the data and label column data
data = df.iloc[:,1:-1]
label = df.iloc[:,-1]

# data = np.array(data).reshape(-1, 53, 12)
data = np.array(data).reshape(-1, 12, 12)
label = np.array(label).reshape(-1, 1)

# split the data and label to train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

# because of TensorFlow to train model, I can not get model medium layer output,
# So I convert to use the original Keras to train the model
import keras
from keras.layers import LSTM, Dense, Dropout,BatchNormalization
from keras.models import Sequential

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
model.add(Dropout(.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(.5))
model.add(LSTM(128))
model.add(Dense(1))
model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adadelta())
model.summary()

his = model.fit(xtrain, ytrain, epochs=100, batch_size=512)
# now we have to get the model's medium layer's outputs
from keras import backend as k
inp = model.inputs
# This we will get all the layers outputs
outputs = [layer.output for layer in model.layers]
functors = [k.function([inp] + [k.learning_phase()], [out]) for out in outputs]
layer_outputs = [func([data, 1.]) for func in functors]
# get what we want layer's outputs
res = np.array(layer_outputs[4])[0,:,:]

# if we just want to get some layer's output
inp = model.inputs
output = model.layers[4].output
functors = k.function([inp]+[k.learning_phase()], [output])
res = functors([np.array(data).reshape(-1, 12, 12), 1.])
res = np.array(res)[0,:,:]




"""
 Because I also want to use the LSTM to train the extracted feature from the orginal dataset,
 I have to set the original dataset model output layer's number be square of some datasets
"""
path = 'F:\workingData\\201806\ALS\\train_data'
df = pd.read_csv(path+'/train_data_cnn_11.csv')

# get the data and label column data
data = df.iloc[:,1:-1]
label = df.iloc[:,-1]
# data = np.array(data).reshape(-1, 53, 12)
data = np.array(data).reshape(-1, 12, 12)
label = np.array(label).reshape(-1, 1)

# This is the new multi-label datasets
new_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
df = pd.read_csv(new_path+'/train_data_12.csv')
df.dropna(inplace=True)
multi_label = df.iloc[:, 146:]

# this is the original training model
def lstm_contruct(last_units = 144):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
    model.add(Dropout(.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.5))
    model.add(LSTM(last_units))
    # this is the output of the regression problem
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adadelta())
    return model

# difine function for the multi-label lstm
def multi_lstm(input_dim, label_dim=multi_label.shape[1]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(input_dim, input_dim)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(label_dim, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta())
    # because of the paper recommend use RMSProp for RNN, this time just use the RMSProp optimizer
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.RMSprop())
    model.summary()
    return model
model_rmsprop = multi_lstm(12)
his_rmsprop = model_rmsprop.fit(xtrain, ytrain, epochs=100, batch_size=256, validation_data=[xtest, ytest])

last_units_list = [144, 169, 225, 400]
def compare_diff_unit_lstm(epochs=100):
    lstm_loss_dict = dict()
    multi_loss_dict = dict()
    multi_acc_dict = dict()
    acc_dict = dict()
    for hiddens in last_units_list:
        model = lstm_contruct(hiddens)
        his = model.fit(data, label, epochs=epochs, batch_size=512)
        lstm_loss_dict[hiddens] = his.history['loss']
        # after we have trained the model, we can get the model outputs
        inp = model.input
        output = model.layers[4].output
        functors = k.function([inp] + [k.learning_phase()], [output])
        res = functors([data, 1.])
        res = np.array(res)[0, :, :].reshape(-1, np.sqrt(hiddens).astype(np.int32),
                                             np.sqrt(hiddens).astype(np.int32))
        # after we have the the extracted features, we can split the extracted features and
        # multi_label
        xtrain_new, xtest_new, ytrain_new, ytest_new = train_test_split(res, np.array(multi_label), test_size=.2, random_state=1234)
        # construct the multi-label model
        mul_model = multi_lstm(np.sqrt(hiddens).astype(np.int32))
        # start to train the multi-label model
        history_mul = mul_model.fit(xtrain_new, ytrain_new, epochs=epochs, batch_size=512)
        multi_loss_dict[hiddens] = history_mul.history['loss']
        multi_acc_dict[hiddens] = history_mul.history['acc']
        acc_dict[hiddens] = mul_model.evaluate(xtest_new, ytest_new, batch_size=1024)

    # return acc_dict, lstm_loss_dict, multi_loss_dict,multi_acc_dict
    print('This is the final model evaluating result:', acc_dict)
    # we can plot the original datasets result and multi-label accuracy and loss
    fig, ax = plt.subplots(3, 1, figsize=(14,12))
    for j in range(len(lstm_loss_dict)):
        ax[0].plot(list(lstm_loss_dict.values())[j], label='Original_last_units_'+np.str(last_units_list[j]))
        ax[0].set_title('Original LSTM Accuracy')
        ax[1].plot(list(multi_acc_dict.values())[j], label='Multi-label of lstm units'+np.str(last_units_list[j]))
        ax[1].set_title('Different units Multi-label Accuracy')
        ax[2].plot(list(multi_loss_dict.values())[j], label='Multi-label of lstm units'+np.str(last_units_list[j]))
        ax[2].set_title('Different units Multi-label Loss')
    plt.legend()
    plt.show()
    # plot the different evaluation accuracy
    plt.scatter(list(acc_dict.keys()), np.array(list(acc_dict.values()))[:,1])
    plt.title('Different units of Test Accuracy')
    plt.show()
    return acc_dict

acc_dict, lstm_loss_dict, multi_loss_dict,multi_acc_dict = compare_diff_unit_lstm()




""" Because I want to plot the different classes curve, so I make it a function"""
new_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
result_data = pd.read_csv(new_path+'/new_best_result.csv')
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def plot_diff_classes(result_data):
    cols = list(result_data.columns)
    index_ch = ['充值类 ','食品类 ','日用品类 ','服装鞋帽类 ','工具材料类 ','家用电器类 ','出版物 ','娱乐旅游类 ','器材设备类 ','非必需品类 ','医药类 ','其它类']
    index_num = np.array(['4 ','4 ','4 ','4 ','5 ','4 ','4 ','4 ','5 ','4 ','5 ','4']).astype(np.int32)
    # loop for each class
    end_index = 0
    for i in range(len(index_ch)):
        print(index_ch[i])
        print('Tttt',end_index)
        tmp_cols = cols[end_index : end_index + index_num[i] * 2]
        x = tmp_cols
        y = np.sum(result_data.ix[:, tmp_cols])
        print(tmp_cols)
        sns.barplot(x=x, y=y)
        plt.title(index_ch[i])
        end_index += index_num[i]*2
        plt.show()
plot_diff_classes(result_data)




""" I have read the NMT papers, for neural machine translation problem,
    it is often better using the bidirectional LSTM to train the model also with BatchNormalization,
    Here I want to use keras to build a Bidirectional LSTM to train on data
"""
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(12, 12)), input_shape=(12, 12)))
model.add(BatchNormalization())
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(.5))
model.add(LSTM(128))
model.add(Dropout(.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(.5))
model.add(Dense(51, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam())







"""
    Because we want to compare with the already loaned persons and no-loaned persons,
    I make it a function for comparing
"""
import math
com_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
df = pd.read_csv(com_path+'/train_data_12.csv')
df2 = pd.read_csv(com_path+'/dueData.csv')
df2.columns = ['over_dues', 'mobile']

sati_df = pd.merge(df, df2, on='mobile', how='inner')

left_join_df = pd.merge(df, df2, on='mobile', how='left')
null_list = list()
for i in range(df.shape[0]):
    null_list.append(math.isnan(left_join_df['over_dues'][i]))
other_df = left_join_df[null_list]

# get the sati and other label for two parts
sati_label = sati_df.iloc[:, 145:]
other_label = other_df.iloc[:, 145:]

# this is the score value for different score index, because I want to sum all the scores for each person
score_cols = np.array(['0','15','25','35','0','15','25','35','0','10','20','30','0','5','10','20','0','2.5','5',
 '7.5','10','0','3','6','10','0','3','6','10','0','3','6','10','0','2.5','5','7.5','10','0',
 '3','6','10','0','2.5','5','7.5','10','0','3','6','10']).astype(np.float32)

sati_all_score = np.sum(np.array(sati_label.iloc[:,:-1]*score_cols), axis=1)
other_all_score = np.sum(np.array(other_label.iloc[:, :-1]*score_cols), axis=1)

sati_all_score_with_overdue = pd.concat((sati_label, pd.DataFrame(sati_all_score.reshape(-1,1), columns=['all_score'])), axis=1)

sns.distplot(sati_all_score, label='loaned')
sns.distplot(other_all_score, label='non-loaned')
plt.legend()
plt.show()


# because of I want to plot the different range overdue ratio, I define a function to achieve it
def plot_diff_range_over_ratio(data=sati_all_score_with_overdue.ix[:,['over_dues','all_score']], ran=10):
    min_v = data['all_score'].min()
    max_v = data['all_score'].max()
    range_v = max_v - min_v
    range_split = int(range_v/ran +1)
    res_dict = dict()
    for i in range(range_split):
        print('This is range: %d'%(min_v+ i*ran))
        # because I want to judge whether the data is between value a and value b, get the satisfied data
        def _judge(d, v1, v2):
            if d >= v1 and d < v2: return True
            else: return False
        sati_list = [_judge(d, min_v + i*ran, min_v + (i+1)*ran) for d in data['all_score']]
        sd = data[sati_list]
        # get all the overdue data
        res_dict[np.str((min_v+i*ran)/2)] = np.float32(sd[sd['over_dues'] != 0].shape[0]/sd.shape[0])
    return res_dict
    # plt.scatter(list(res_dict.keys()), list(res_dict.values()))
    plt.plot(list(res_dict.values()))
    # plt.xticks(list(res_dict.keys()))
    plt.title('Different range plot')
    plt.show()






"""
    This is for the on-line model, abstract all the features to one function:
    One: For using the ALS to train the recommendation problem
    Two: For using LSTM to get the multi-label problem, I can just load the model
    Three: For using LSTM to get the sum-score for each person problem
"""
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# This function is to recommendate the users' implicit features
def als_recommend(data=None):
    conf = SparkConf.setAppName('ALS_Recommendation')
    spark = SparkSession.builder.config(conf).enableHiveSupport().getOrCreate()
    # if the data is not given, before I just read Mysql database
    if data is None:
        connection = pymysql.connect(user='zhanghui', password='zhanghui', database='model_data', host='10.1.36.18',
                                     charset='utf8')
        # query = 'select mobile , label2, count(1) as rating from (select *  from model_data.tb_item_classified where label2 is not null and price !=0 and status != "已取消")t group by mobile, label2'
        query2 = "select mobile, label1, count(1) from (select * from model_data.tb_item_classified where label2 is not null and price > 0 and status !='已取消')t group by mobile, label1"
        re = pd.read_sql(query2, con=connection)
        re.columns = ['userId_raw', 'itemId_raw', 'rating']
    else:
        # read the Hive warehouse
        pass
    # This is assumed that the readed DataFrame name is re
    df_read = spark.createDataFrame(re)

    # Because the original data is just String label for each class,
    # Here I use the StringIndexer to index it
    # Because I have to recommend for all the new data with the original data,
    # so the StringIndexer have to be trained
    df_userid = StringIndexer(inputCol='userId_raw', outputCol='userId').fit(df_read).transform(df_read)
    df = StringIndexer(inputCol='itemId_raw', outputCol='itemId').fit(df_userid).transform(df_userid)

    als = ALS(maxIter=3, regParam=1, rank=10, itemCol='itemId', ratingCol='rating', userCol='userId',
              coldStartStrategy='drop')
    model = als.fit(df)
    pred = model.transform(df)
    return pred


# This function is used to prediction the new come person for multi-label implement,
# For whether or not the indexed score columns appeared
import tensorflow as tf

def multi_label_classification(data=None):
    if data is None:
        new_path = 'F:\workingData\\201806\\recommendation\MultiLabel'
        df = pd.read_csv(new_path + '/train_data_12.csv')
        df.dropna(inplace=True)
        data = df.iloc[:, :145].drop('mobile', axis=1)
        label = df.iloc[:, 145:]
        # So the data must be size 144
        data = np.asarray(data).reshape(-1, 12, 12)
    else:
        pass
    # load the already trained model
    model_path = 'F:\workingData\\201806\\recommendation\MultiLabel\multi_model'
    model = tf.keras.models.load_model(model_path + '/LSTM_multi_class.h5')
    pred = model.predict(data)
    f = lambda x:1 if x>=.5 else 0
    pred = pd.DataFrame(pred).applymap(f)
    # return pred is DataFrame that means for whether or not the indexed score column appeared
    return pred


# This function is used to predict the sum-score for each person, using LSTM model to predict
def score_pred(data = None):
    if data is None:
        path = 'F:\workingData\\201806\ALS\\train_data'
        df = pd.read_csv(path + '/train_data_cnn_11.csv')

        # get the data and label column data
        mobile = df.iloc[:, 0]
        data = df.iloc[:, 1:-1]
        label = df.iloc[:, -1]
        # data = np.array(data).reshape(-1, 53, 12)
        # This data is batch_size*12*12
        data = np.array(data).reshape(-1, 12, 12)
        label = np.array(label).reshape(-1, 1)
    else:
        pass
    model_path = 'F:\workingData\\201806\\recommendation\MultiLabel\multi_model'
    model = tf.keras.models.load_model(model_path + '/LSTM_regression.h5')
    # this is the all the needed prediction score
    pred = model.predict(data)
    # if I have getted the mobile, just combine the mobile and prediction two columns
    out = np.concatenate((np.array(mobile).reshape(-1,1), pred), axis=1)
    return out
