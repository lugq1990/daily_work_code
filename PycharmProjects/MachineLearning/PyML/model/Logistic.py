# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:57:51 2017

@author: Administrator
"""
import pandas as pd
import numpy as np

from scipy.optimize import minimize


class Logit():
    
    def __init__(self):
        self.X = 0
        self.y = 0
        self.y0 = 0
        self.y1 = 0
        self.feat_names = None
        self.w = 0
        self.result = 0
    
    #逻辑回归模型的对数似然函数
    def _logit(self, w):
        w0 = w[0]
        wx = w[1:]
        z = np.dot(self.X, wx) + w0
        p = 1 / (1 + np.exp(-z))
        LL = np.sum(self.y1 * np.log(p).T + self.y0 * np.log(1-p).T)
        return -1 * LL
    
    #逻辑回归模型参数的雅克比向量
    def _logit_f(self, w):
        w0 = w[0]
        wx = w[1:]
        z = np.dot(self.X, wx) + w0
        p = 1 / (1 + np.exp(-z))
        XX = np.c_[np.ones(self.X.shape[0]), self.X]
        return -np.array(np.dot((self.y - p), XX).tolist()[0])
    
    def fit(self, X, y, feat_names):
        self.feat_names = feat_names
        self.w = np.random.rand(X.shape[1] + 1) * 0.1
        self.X = X
        self.y = y
        self.y0 = np.zeros(y.shape)
        self.y1 = np.zeros(y.shape)
        self.y0[y==0] = 1
        self.y1[y>0] = 1
        #优化计算
        self.opt = minimize(self._logit, self.w, method='BFGS', jac=self._logit_f, 
                            options={'gtol': 1e-6, 'maxiter': 5000, "disp": True})
    
    def print_result(self):
        LL0 = -self._logit(self.w)
        LL1 = -self._logit(self.opt.x)
        rho = 1-LL1/LL0
        rho2 = 1-(LL1-len(self.opt.x))/LL0
        
        #计算参数误差
        err = np.sqrt(np.abs(np.diag(self.opt.hess_inv)))
        #计算参数的t值
        t_stat = self.opt.x/np.sqrt(np.abs(np.diag(self.opt.hess_inv)))
        
        fea = ['constant']
        fea.extend(self.feat_names)
        #print u"参数初始值:"
        #print self.w
        #print
        #打印模型的参数估计结果
        result = pd.DataFrame({'feature_name': fea, 'value': self.opt.x, 'error': err, 't_score': t_stat}, 
                              columns=['feature_name', 'value', 'error', 't_score'])
        result['t_abs'] = np.abs(result['t_score'])
        result0 = result.iloc[0:1, :]
        result1 = result.iloc[1:, :].sort_values(by='t_abs', ascending=False)
        result = pd.concat([result0, result1], axis=0, ignore_index=True)
        self.result = result.iloc[:, :4]
        print u"------------------参数估计结果------------------"
        print self.result
        print "------------------------------------------------"
        print "LL0: %.3f  LL1: %.3f  rho: %.3f  rho2: %.3f" %(LL0, LL1, rho, rho2)

    def get_feat_names(self, n):
        return self.result[np.abs(self.result['t_score'])>=n]['feature_name'].tolist()
