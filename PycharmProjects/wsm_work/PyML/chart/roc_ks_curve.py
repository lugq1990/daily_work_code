# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 13:05:53 2017

@author: Administrator
"""
import matplotlib.pyplot as plt

from scikitplot import plotters as skplt
from sklearn.metrics import roc_curve, auc


#=================================== 作图 ======================================
#ks chart
def DRAW(y_true, y_probas):
    skplt.plot_ks_statistic(y_true=y_true, y_probas=y_probas)
    plt.show()
    
    false_positive_rate, recall, thresholds = roc_curve(y_true, y_probas[:, 1])
    roc_auc = auc(false_positive_rate, recall)  
    plt.title('ROC curve of validation (AUC=%0.4f)'% roc_auc)
    plt.plot(false_positive_rate, recall, 'b')
    plt.grid(linestyle='--', linewidth=.5)
    plt.legend(loc='lower right')  
    plt.plot([0, 1], [0, 1], 'r--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.0])  
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')  
    plt.show() 




