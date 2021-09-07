# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import tensorflow as tf


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
df = df_read#.sample(n=10000)

data = np.array(df.drop(['91'],axis=1))
label = np.array(df['91'])#.reshape(-1,1)
#split the data for train and test
xtrain,xtest,ytrain,ytest = train_test_split(data,label,test_size=.3)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':xtrain},y=ytrain,num_epochs=None,shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':xtest},y=ytest,num_epochs=1,shuffle=False)

feature_columns = [tf.feature_column.numeric_column('x',shape=[data.shape[1]])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[512,1024,256],
                                         n_classes=2,optimizer=tf.train.AdamOptimizer(learning_rate=1e-4))
classifier.train(input_fn=train_input_fn,steps=10000)
#compute the accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
print("the test accuracy is ",accuracy_score)
pred = classifier.predict(input_fn=test_input_fn)
tmp = np.array(tuple(pred))
out = np.empty((tmp.shape[0]))
for i in range(tmp.shape[0]):
    out[i] = tmp[i]['probabilities'][1]

#evaluate the model
from sklearn import  metrics
auc = metrics.roc_auc_score(ytest,out)
fpr,tpr,_ = metrics.roc_curve(ytest,out)
ks = tpr - fpr
pred = np.array(pd.Series(out).map(lambda x:1 if x>=.5 else 0))
recall = metrics.recall_score(ytest,pred)
f1 = metrics.f1_score(ytest,pred)
confusion = metrics.confusion_matrix(ytest,pred)
print('auc=',auc)
print('ks=',ks.max())
print('recall=',recall)
print('f1 score=',f1)
print('confusion matrix',confusion)
