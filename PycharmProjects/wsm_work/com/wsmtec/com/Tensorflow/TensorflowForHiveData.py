# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics

t1 = time.time()
path = 'F:\workingData\\201709\data\Hive'
df_read = pd.read_csv(path+'/test.txt',sep='\t')
shuffle(df_read)
#add the colums
tmp = np.arange(1,94)
col = list()
for i in range(tmp.shape[0]):
    col.append(np.str(tmp[i]))
df_read.columns = col
df_read.drop(['1','92','93'],axis=1,inplace=True)
df = df_read

data = np.array(df.drop(['91'],axis=1))
label = np.array(df['91']).reshape(-1,1)
#split the data for train and test
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.3)

#construct the tensorflow model
import tensorflow as tf
learning_rate = 1e-4
batch_size = 128
training_epoch = 10
display_step = 100
n_hidden_1 = 512
n_hidden_2 = 1024
n_features = data.shape[1]
drop = .5

x = tf.placeholder(tf.float32,[None,n_features])
y = tf.placeholder(tf.float32,[None,1])
dropout = tf.placeholder(tf.float32)

weights = {'h1':tf.Variable(tf.random_normal([n_features,n_hidden_1])),
          'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
          'out':tf.Variable(tf.random_normal([n_hidden_2,1]))}
biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
         'b2':tf.Variable(tf.random_normal([n_hidden_2])),
         'out':tf.Variable(tf.random_normal([1]))}

def mlp(x,weights,biases):
    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)
    #add the dropout, for every layer add a dropout layer for nomalization
    layer1 = tf.nn.dropout(layer1,drop)
    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)
    layer2 = tf.nn.dropout(layer2,drop)
    out = tf.add(tf.matmul(layer2,weights['out']),biases['out'])
    out = tf.nn.sigmoid(out)
#     out = tf.cast(out,tf.int64)
    return out

pred = mlp(x,weights=weights,biases=biases)
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
#run the tensorflow
with tf.Session() as sess:
    sess.run(init)
    for i in range(15):
        num_train = xtrain.shape[0]
        avg_cost = 0.
        _,c = sess.run([optimizer,loss],feed_dict={x:xtrain,y:ytrain,dropout:drop})
        avg_cost += c/num_train
        prediction = sess.run(pred,feed_dict={x:xtrain,y:ytrain,dropout:drop})
        for j in range(prediction.shape[0]):
            if prediction[j][0] >= .5:
                prediction[j][0] = 1
            else:
                prediction[j][0] = 0
        #print("!!!",prediction.astype(np.int64))
        prediction = prediction.astype(np.int64)
        #print("pred ",prediction)
        correct_pred = np.equal(prediction,ytrain)
        #print("!!",correct_pred)
        accuracy = np.sum(correct_pred.astype(np.float32))
        acc = accuracy/num_train
        print("the epoch is %2d" %(i+1)," and the train acc is ","{:.5f}".format(acc)," and the loss is ","{:.5f}".format(avg_cost))
#        if i % 2 == 0:
#            print("the epoch is %d"%(i+1)," the average cost is ","{:.7f}".format(avg_cost))
#            print("pred ",np.argmax(prediction,1))
    print("the optimization is done!")
    num_test = xtest.shape[0]
    test_cost = sess.run(loss,feed_dict={x:xtest,y:ytest,dropout:drop})
    test_avg_cost = test_cost/num_test
    prediction = sess.run(pred,feed_dict={x:xtest,y:ytest,dropout:drop})
    for j in range(prediction.shape[0]):
        if prediction[j][0] >= .5:
            prediction[j][0] = 1
        else:
            prediction[j][0] = 0
    prediction = prediction.astype(np.int64)
    correct_pred = np.equal(prediction,ytest)
    accuracy = np.sum(correct_pred.astype(np.float32))
    acc = accuracy/num_test
    print("the test loss is ",test_avg_cost," and the test acc is ","{:.5f}".format(acc))

