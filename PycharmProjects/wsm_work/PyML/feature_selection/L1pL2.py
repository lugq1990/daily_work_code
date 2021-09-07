# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:15:16 2017

@author: Administrator
"""
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.base import clone


class FeatureSelectionL1L2():
    
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self._threshold = threshold
        self.columns = None
        self.l1 = clone(LogisticRegression(penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs))
        
        #使用同样的参数创建L2逻辑回归
        self.l2 = clone(LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C,
                                     fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                     class_weight = class_weight, random_state=random_state, solver=solver, 
                                     max_iter=max_iter, multi_class=multi_class, verbose=verbose, 
                                     warm_start=warm_start, n_jobs=n_jobs))

    def fit(self, X, y):
        #训练L1逻辑回归
        self.l1.fit(X, y)
        #训练L2逻辑回归
        self.l2.fit(X, y)
        cntOfRow, cntOfCol = self.l1.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.l1.coef_[i, j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i, j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i, k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self._threshold and j != k and self.l1.coef_[i, k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.l1.coef_[i, idx] = mean
        rng = np.arange(cntOfCol)
        self.columns = rng[list(self.l1.coef_[0]!=0)]
        return X[:, list(self.l1.coef_[0]!=0)]


#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
if __name__=="__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    data = pd.read_csv(r"E:\modeling\data\new_data1_1.csv")
    y = np.array(data.overdue_days)
    y[y>0] = 1
    X = data.iloc[:,2:40]
    X = StandardScaler().fit_transform(X)
    lr = FeatureSelectionL1L2(threshold=0.001, C=0.01)
    ret = lr.fit(X, y)
    print X.shape
    print ret
    print lr.columns
    
