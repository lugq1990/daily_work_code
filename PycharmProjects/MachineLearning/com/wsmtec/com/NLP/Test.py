# -*- coding:utf-8 -*-
# import io
#
# path ='E:\ExaData\\fastText/'
# file_name = 'cc.zh.300.vec'
#
# def load_vector(fname):
#     fin = io.open(fname,'r',encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data
#
# data = load_vector(path+ file_name)
# print(data)

from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import scikitplot as skplt

iris = load_iris()
x, y = iris.data, iris.target
"if you want to get the probability, set probability to be True"
clf = SVC(probability=True)
clf.fit(x, y)
pred = clf.predict(x)
prob = clf.predict_proba(x)
"""you can plot the precision_recall_curve and confusion matrix"""
skplt.metrics.plot_precision_recall_curve(y, prob)
skplt.metrics.plot_confusion_matrix(y, pred)
plt.show()

"""
Here is for LinearSVC
first way is to set the kernel of SVM to be 'linear' to replace the LinearSVC
"""
clf = SVC(probability=True, kernel='linear')
clf.fit(x, y)
"others is same"

"""second way is use the CalibratedClassifierCV for more generally useful
for that not support predict_proba"""
svm = LinearSVC()
clf_cal = CalibratedClassifierCV(svm)
clf_cal.fit(x, y)
pred = clf_cal.predict(x)
prob = clf_cal.predict_proba(x)
