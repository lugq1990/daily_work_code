# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
import logging
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras import optimizers

path = 'F:\workingData\\201806\\recommendation\MultiLabel'

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

connection = pymysql.connect(user='zhanghui', password='zhanghui', database='model_data', host='10.1.36.18',
                             charset='utf8')

query = """select mobile,sum(recharge) as t1, sum(food) as t2, sum(commodity) as t3,
sum(clothing) as t4, sum(tools) as t5, sum(elect) as t6,  sum(publish) as t7, sum(entert) as t8,
sum(equip) as t9, sum(noness) as t10, sum(medicine) as t11, sum(other) as t12
from model_data.train_data_X_12 group by mobile, day_no"""

item_df = pd.read_sql(query, con=connection)

org_df = pd.read_csv(path+'/train_data_12.csv')
needed_label = ['mobile','rechg_0','rechg_1','rechg_2','rechg_3','foo_0','foo_1','foo_2','foo_3','comdy_0','comdy_1','comdy_2','comdy_3','clo_0','clo_1','clo_2','clo_3','tols_0','tols_1','tols_2','tols_3','tols_4','ele_0','ele_1','ele_2','ele_3','pub_0','pub_1','pub_2','pub_3','ent_0','ent_1','ent_2','ent_3','eqp_0','eqp_1','eqp_2','eqp_3','eqp_4','ness_0','ness_1','ness_2','ness_3','med_0','med_1','med_2','med_3','med_4','oth_0','oth_1','oth_2','oth_3']

need_df = org_df[needed_label].dropna()
need_df['mobile'] = need_df['mobile'].astype(np.str)

# Join the tow dataframe, because of the pandas dataframe join must specified the index columns,
# make the need_df index as mobile, left join.
res_df = item_df.join(need_df.set_index('mobile'), on='mobile',how='left').dropna()

# then convert the res_df to be three parts: mobile, data and multi-label
mobile = res_df['mobile'].unique()
data = res_df.iloc[:, 1: 13]
label_org = np.array(res_df.iloc[:, 13:])

# we have to do 2 things:
# 1.convert the data to be 3-D(n, 12, 12);
# 2. get the each label for each item,
# because for now the label dataframe is for each person will be 12 rows

# convert the data to n*12*12
data = np.array(data).reshape(-1, 12, 12)

# Because of the not extremely better result, maybe use data.T will be better for LSTM?
new_data = np.empty_like(data)
for i in range(data.shape[0]):
    new_data[i, :, :] = data[i, :, :].T


# get the label use the numpy.empty
label = np.empty((mobile.shape[0], label_org.shape[1]), dtype=np.int64)
j = 0
for i in range(len(label_org)):
    if i % 12 == 0:
        label[j, :] = label_org[j*12, :]
        j += 1

# before we train the model, we have to split the data to train and test datasets
xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)


# now we have  already get the training data and label, we can define our model, have fun.
# Here I use the Stacked LSTM to build the model
# lstm->dropout->lstm->drouout->lstm->dropout->dense

inputs = Input(shape=(12, 12))
lstm_1 = LSTM(128, return_sequences=True)(inputs)
lstm_1 = Dropout(.5)(lstm_1)
lstm_2 = LSTM(128, return_sequences=True)(lstm_1)
lstm_2 = Dropout(.5)(lstm_2)
lstm_3 = LSTM(128)(lstm_2)
lstm_3 = Dropout(.5)(lstm_3)
out = Dense(label.shape[1], activation='sigmoid')(lstm_3)
model = Model(inputs, out)


print('Model parameters number for each layer:')
model.summary()

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizers.Adadelta())

# define the early-stopping for the model
es = EarlyStopping(monitor='val_loss', patience=3)

his = model.fit(xtrain, ytrain, epochs=50, verbose=1, batch_size=1024, validation_data=(xtest, ytest), callbacks=[es])







