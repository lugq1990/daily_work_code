# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import time
import keras
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,GlobalAveragePooling1D,BatchNormalization,Input,LSTM
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from sklearn.utils import class_weight
from keras.layers import Activation,Flatten
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_curve
import lightgbm as lgb
import gc

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def score(y, pred, pos_label=1):
    fpr, tpr, thre = roc_curve(y, pred, pos_label=pos_label)
    score = 0.4*tpr[np.where(fpr>=0.001)[0][0]] + 0.3*tpr[np.where(fpr>=.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]]
    return score

path = 'F:\workingData\Compitition\\alipay'
df = pd.read_csv(path+'/atec_anti_fraud_train.csv', error_bad_lines=False)
df_test = pd.read_csv(path+'/atec_anti_fraud_test_a.csv', error_bad_lines=False)

# get the satisfied data, not too much Non columns
df_read = df
tol = 500000
sati_cols = df_read.columns[df_read.isnull().sum() <= tol]
df_read = df_read[sati_cols]

df_read = df_read[df_read['label']!=-1]
data = df_read.drop(['id','label','date'], axis=1)
label = df_read['label']

# get all the columns' mean value
mean_value = data.mean()
""" use smote to generate negative data"""
# from com.wsmtec.com.MachineLearningAlgorithm.SMOTE_2 import SMOTE
# def gen_data():
#     neg_data = df_read[df_read['label'] == 1].drop(['id','label','date'], axis=1)
#     pos_data = df_read[df_read['label'] == 0].drop(['id','label','date'], axis=1)
#     neg_label = df_read['label'][df_read['label'] ==1]
#     pos_label = df_read['label'][df_read['label'] == 0]
#     neg_data = neg_data.fillna(mean_value)
#     pos_data = pos_data.fillna(mean_value)
#     smote = SMOTE(np.array(neg_data), N=200)
#     re_smote = smote.over_sample()
#     neg_label_new = pd.concat((neg_label, neg_label), axis=0)
#     re_data = np.concatenate((np.array(pos_data), re_smote), axis=0)
#     re_label = np.concatenate((np.array(pos_label),np.array(neg_label_new)), axis=0)
#     return re_data, re_label

test_data = df_test[sati_cols.drop(['label','id'])].fillna(mean_value)
id_data = np.array(df_test['id'])
data_pred = np.array(test_data.iloc[:,1:])

# fill all the non data
data = data.fillna(mean_value)

xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.1, random_state=1234)

class_weight = class_weight.compute_class_weight('balanced',np.unique(ytrain),ytrain.reshape(-1,))
# add the class_weight for the model

# history = model.fit(xtrain, ytrain, epochs=10, batch_size=1024,class_weight=class_weight)

data_lstm = np.array(data).reshape(-1, 3, 95)
xtrain_lstm, xtest_lstm, ytrain_lstm, ytest_lstm = train_test_split(data_lstm, np.array(label).reshape(-1,1),
                                                                    test_size=.1, random_state=1234)
del data_lstm

def lstm_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(xtrain_lstm.shape[1],xtrain_lstm.shape[2])))
    model.add(Dropout(.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.5))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['acc',auc_roc], optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model
model_lstm = lstm_model()
# history_lstm = model_lstm.fit(xtrain_lstm, ytrain_lstm, epochs=20, batch_size=1024, class_weight=class_weight)

def res_input_model(num_layers=8, fc_needed=True, fc_units=512, needed_dropout=True,
                    dropout=.5, data=xtrain_lstm, batchnorm=True,
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

    pred = Dense(1, activation='sigmoid')(layer)

    model = Model(inputs=[inputs], outputs=pred)
    model.compile(loss='binary_crossentropy', metrics=['accuracy',auc_roc], optimizer=keras.optimizers.Adadelta())

    print('Model Struture:')
    model.summary()

    return model
model_res_input_8 = res_input_model()
#history_res_input_8 = model_res_input_8.fit(xtrain_lstm, ytrain_lstm, epochs=20, batch_size=1024, class_weight={0:.1,1:100})

def train_model(model, epochs=20, batch_size=1024, class_weight=class_weight):
    s_t = time.time()
    history = model.fit(xtrain_lstm, ytrain_lstm, epochs=epochs,batch_size=batch_size, class_weight=class_weight)
    e_t = time.time()
    print('Total time is %.2f seconds'%(e_t - s_t))
    return history

def eval_model(model, batch_size=4096):
    acc = model.evaluate(xtest_lstm, ytest_lstm, batch_size=batch_size)[1]
    prob = model.predict(xtest_lstm)
    pred_score = score(ytest_lstm, prob)
    print('Model Accuracy = %.4f'%(acc))
    print('Model Score = ',(pred_score))




""" use lightGBM to train the data, I find that not num_leaves is larger, the score is higher.
    It is important to set the early_stopping to 5 or 10 for example
    Notes: scale_pos_weight set to 1000 is bad,
          is_unbalance=True, score:0.63,
          basic params: acore:0.6695,
          boosting_type：'dart', score:0.63, using lr=0.01 score is lower:0.59
          num_leaves=400,lr=.05: best score:0.6935
    num_leaves = 100 is best for generation
"""
lgb_train = lgb.Dataset(xtrain, ytrain, free_raw_data=False)
lgb_test = lgb.Dataset(xtest, ytest, free_raw_data=False)

# get the categorical columns of the original dataset for training time, tell the
# model which columns is categorical
col_type = data.dtypes
cate_cols = col_type.index[col_type == np.int64]

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc','binary_logloss'],
    'num_leaves': 100,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': 1
    #'weight':class_weight,
    #'is_unbalance':True
}

# gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test)
# gbm_score = score(ytest, gbm.predict(xtest))
# print('BGM score ',gbm_score)

""" make the final prob to output DataFrame"""
def save_result(prob, file_name=None):
    out = np.concatenate((id_data, prob), axis=1)
    out = pd.DataFrame(out, columns=['id', 'score'])
    out.to_csv(path+'/prediction/%s'%file_name, index=False)

""" define a function to evaluate different num_leaves influence the accuracy"""
param_list = [150, 200, 250]
def compare(param, param_list=param_list):
    model_list = list()
    score_list = list()
    evals_list = list()
    for l in param_list:
        evals_result = {}
        params[param] = l
        model = lgb.train(params, lgb_train, num_boost_round=300, valid_sets=[lgb_train,lgb_test], evals_result=evals_result)
        model_list.append(model)
        evals_list.append(evals_result)
        ss = score(ytest, model.predict(xtest))
        score_list.append(ss)
    return model_list, evals_list, score_list
model_list, evals_list, score_list = compare('learning_rate',[.1, .05, .01])


""" use One-Hot encoder for all the categorical columns to be one-hot"""
col_type = data.dtypes
cate_cols = col_type.index[col_type == np.int64]
cate_data = data[cate_cols]
from sklearn.preprocessing import OneHotEncoder
data_one = pd.DataFrame(OneHotEncoder().fit_transform(cate_data).toarray())
del cate_data
gc.collect()

""" just for testing, making the float to int"""
conti_cols = col_type.index[col_type == np.float64]
conti_data = data[conti_cols].astype(np.int32)
data_conti = np.concatenate((data[cate_cols], conti_data), axis=1)
del conti_data
# combine the one-hot encoder's columns and the original data
data_new = np.concatenate((data, data_one), axis=1)
xtrain_new, xtest_new, ytrain_new, ytest_new = train_test_split(np.array(data_conti), np.array(label),
                                                                test_size=.1, random_state=1234)
del data_new, data_conti
gc.collect()

lgb_train_new = lgb.Dataset(xtrain_new, ytrain_new)
lgb_test_new = lgb.Dataset(xtest_new, ytest_new, reference=lgb_train_new)

# define the function to train the model
def train_gbm_new(params, lgb_train=lgb_train_new, lgb_test=lgb_test_new, nums_steps=200, early=30):
    evals_result = {}
    gbm = lgb.train(params, lgb_train, num_boost_round=nums_steps, valid_sets=[lgb_train, lgb_test],
                    evals_result=evals_result, early_stopping_rounds=early)
    prob = gbm.predict(xtest_new)
    ss = score(ytest_new, prob)
    # prob_n = prob.reshape(-1, 1)
    # skplt.metrics.plot_roc_curve(ytest_new, np.concatenate((1 - prob_n, prob_n), axis=1))
    pred = pd.Series(prob).apply(lambda x:1 if x>=.5 else 0)
    skplt.metrics.plot_confusion_matrix(ytest_new, pred)
    lgb.plot_metric(evals_result, metric='auc')
    # lgb.plot_metric(evals_result, metric='binary_logloss')
    plt.show()
    print('Model score = %.6f'%(ss))
    return gbm
model = train_gbm_new(params)



def train_gbm(params, lgb_train=lgb_train, lgb_test=lgb_test, num_steps=300, early=30):
    s_t = time.time()
    evals_result = {}
    gbm = lgb.train(params, lgb_train, num_boost_round=num_steps, valid_sets=[lgb_train, lgb_test],
                    evals_result=evals_result, early_stopping_rounds=early,
                    categorical_feature=list(cate_cols),
                    verbose_eval=True)
    prob = gbm.predict(xtest)
    e_t = time.time()
    ss = score(ytest, prob)
    prob_n = prob.reshape(-1, 1)
    # skplt.metrics.plot_roc_curve(ytest_new, np.concatenate((1 - prob_n, prob_n), axis=1))
    pred = pd.Series(prob).apply(lambda x:1 if x>=.5 else 0)
    skplt.metrics.plot_confusion_matrix(ytest, pred)
    lgb.plot_metric(evals_result, metric='auc')
    # lgb.plot_metric(evals_result, metric='binary_logloss')
    plt.show()
    print('Model score = %.6f'%(ss))
    print('Model times = :'+np.str(e_t - s_t))
    return gbm


"""There is the other person uses params for unbalanced data:

params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced
}

"""

""" according to the lightGBM weight file, just make a random weight for each item,
    using a random gaussian weight, shuffle it and make it into a weighted-file to disk for lightGBM，
    the weight of lightGBM is must be 1-D array or just a list.
 """
rand_weight = np.random.random(size=(data.shape[0]))
from sklearn.utils import shuffle
rand_weight = shuffle(rand_weight)
rand_weight = pd.DataFrame(rand_weight)
rand_weight.to_csv(path+'/weight.csv', index=False, header=False)



""" using the over-sample algorithm to generate new negative data and
combine all of them and return to the train and test data
"""
# from com.wsmtec.com.MachineLearningAlgorithm.My_over_sampling import my_over_sampling
# def gen_neg(nums = 500):
#     df_tmp = df_read.drop(['id', 'date'], axis=1)
#     neg_data = df_tmp[df_tmp['label'] == 1]
#     generate_neg_data = my_over_sampling(np.array(neg_data.drop(['label'], axis=1)), num=nums).over_sampling()
#     generate_neg_data = pd.DataFrame(generate_neg_data, columns=neg_data.drop(['label'], axis=1).columns)
#     generate_neg_label = pd.DataFrame(np.ones(nums))
#     new_df = pd.concat((df_tmp.drop(['label'], axis=1), generate_neg_data), axis=0)
#     new_label = pd.concat((df_read['label'], generate_neg_label), axis=0)
#     xtrain_gen, xtest_gen, ytrain_gen, ytest_gen = train_test_split(new_df, new_label.reshape(-1,),
#                                                                     test_size=.1, random_state=1234)
#     # del df_tmp, neg_data, generate_neg_label, generate_neg_label,new_df,new_label
#     gc.collect()
#     return xtrain_gen, xtest_gen, ytrain_gen, ytest_gen
# xtrain_gen, xtest_gen, ytrain_gen, ytest_gen = gen_neg()
# ytrain_gen = np.array(ytrain_gen).reshape(-1,).astype(np.int32)
# ytest_gen = np.array(ytest_gen).reshape(-1,).astype(np.int32)
# lgb_train = lgb.Dataset(xtrain_gen, ytrain_gen, free_raw_data=False)
# lgb_test = lgb.Dataset(xtest_gen, ytest_gen, free_raw_data=False, reference=lgb_train)

