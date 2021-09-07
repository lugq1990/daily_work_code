# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

path = 'F:\workingData\\201801\model_evaluation'
df = pd.read_csv(path+'/test.csv')
neg_prob = np.array(df)[:,2].astype(np.float)
label = np.array(df)[:,3].astype(np.int)

from sklearn import metrics
import scikitplot as skplt

pos_prob = (1-neg_prob)
prob = np.concatenate((pos_prob.reshape(-1,1),neg_prob.reshape(-1,1)),axis=1)

pred = np.array(pd.Series(neg_prob).map(lambda x:1 if x>=.5 else 0))

skplt.metrics.plot_confusion_matrix(label,pred)
skplt.metrics.plot_ks_statistic(label,prob)
skplt.metrics.plot_roc_curve(label,prob)

acc = metrics.accuracy_score(label,pred)
print('accuracy is ',acc)
# plt.show()
recall = metrics.recall_score(label,pred)
precision = metrics.precision_score(label,pred)
f1_score = metrics.f1_score(label,pred)
auc = metrics.roc_auc_score(label,neg_prob)
fpr,tpr,_ = metrics.roc_curve(label,neg_prob)
ks = (tpr-fpr).max()
confusion = metrics.confusion_matrix(label,pred)

print("recall = %f"%recall)
print("precision = %f"%precision)
print("f1_score = %f"%f1_score)
print("ks = %f"%ks)
print("recall = %f"%recall)
print("confusion matrix is ")
print(confusion)


