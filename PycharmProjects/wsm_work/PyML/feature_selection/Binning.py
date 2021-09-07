# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:59:57 2017

@author: Administrator
"""
import pandas as pd
import numpy as np


class SplitBins():
    
    def __init__(self, max_iter=10, verbose=False, labels=None, retbins=False, precision=3, duplicates='raise', zerobin=False, woe=False):
        self._max_iter = max_iter
        self._labels = labels
        self._retbins = retbins
        self._precision = precision
        self._duplicates = duplicates
        self._zerobin = zerobin
        self.woe = woe
        self.verbose = verbose
        self.IV_table = None

    def _initializer(self):
        self._splitedbins = pd.DataFrame()
        self._ret = {}
        self._ret["feature"] = []
        self._ret["iv"] = []
        self._ret["ks"] = []
        self._ret["bins"] = []
        self._ret['woe'] = []
    
    def _cut(self, x, q):
        labels = self._labels
        retbins = self._retbins
        precision = self._precision
        duplicates = self._duplicates
        name = x.name
        if self._zerobin:
            df = pd.DataFrame({name: [pd.Interval(-999., 0.0, closed='right')] * x.shape[0]})
            df[name][x>0] = pd.qcut(x[x>0], q, labels=labels, retbins=retbins, precision=precision, duplicates=duplicates)
        else:
            df = pd.qcut(x, q, labels=labels, retbins=retbins, precision=precision, duplicates=duplicates)
        return df

    def fit(self, X, y):
        self._initializer() 
        df = pd.DataFrame({y.name: y})
        target = y.name
        ubound = self._max_iter + 5
        col_names = X.columns
        for col_name in col_names:
            name = col_name + "_bins"
            best_ks = 0
            best_iv = 0
            best_bins = 0
            best_splited = None
            for b in range(4, ubound):
                df[name] = self._cut(X[col_name], b)
                tbl = df[[target, name]].groupby(name).count() * 1.
                tbl["yes_overdue"] = df[[target, name]].groupby(name).sum() * 1.
                tbl["no_overdue"] = tbl[target] - tbl["yes_overdue"]
                tbl["DB"] = tbl["no_overdue"].apply(lambda x: x / tbl.no_overdue.sum())
                tbl["DB_cum"] = np.cumsum(tbl["DB"])
                tbl["DG"] = tbl["yes_overdue"].apply(lambda x: x / tbl.yes_overdue.sum())
                tbl["DG_cum"] = np.cumsum(tbl["DG"])
                tbl["woe"] = np.log(tbl["DB"] / tbl["DG"])
                tbl["iv"] = (tbl["DB"] - tbl["DG"]) * tbl["woe"]
                tbl["iv"] = tbl["iv"].replace(np.inf, 0.)
                ks = np.round(np.max(np.abs(tbl["DB_cum"] - tbl["DG_cum"])), 4)
                iv = np.round(np.sum(tbl["iv"]), 4)
                if iv > best_iv and ks > best_ks:
                    best_iv = iv
                    best_ks = ks
                    if self._zerobin: b += 1
                    best_bins = b
                    best_splited = df[name].copy()
                    woe_dict = tbl["woe"].copy()
            self._save_result(col_name, best_iv, best_ks, best_bins, woe_dict)
            if self.woe:
                self._splitedbins[name] = self._replace_woe(best_splited, woe_dict)
            else:
                self._splitedbins[name] = best_splited
        if self.verbose: self._print_result()
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self._splitedbins
    
    def transform(self, X):
        columns = X.columns
        transformed = pd.DataFrame()
        def fit_interval(x, inter_dict=None):
            round_x = round(x, self._precision)
            idx = inter_dict.index
            rng = range(1, (len(inter_dict)-1))
            for i in rng:
                if i == 1:
                    if round_x <= idx[i].left:
                        return inter_dict[inter_dict.index[i - 1]]
                elif i == rng[len(rng)-1]:
                    if round_x > idx[i].right:
                        return inter_dict[inter_dict.index[i + 1]]
                if idx[i].left < round_x <= idx[i].right:
                    return inter_dict[inter_dict.index[i]]
        
        for col in columns:
            name = col + "_bins"
            transformed[name] = X[col].apply(fit_interval, inter_dict=self._ret["woe"][self._ret["feature"].index(col)])
        return transformed
        
    def _replace_woe(self, tbl_left, tbl_right):
        return tbl_left.apply(lambda x: tbl_right[x], self._ret["woe"])
        
    def _save_result(self, col_name, iv, ks, bins, woe_dict):
        self._ret["feature"].append(col_name)
        self._ret["iv"].append(iv)
        self._ret["ks"].append(ks)
        self._ret["bins"].append(bins)
        self._ret["woe"].append(woe_dict)
        
    def _print_result(self):
        self.IV_table = pd.DataFrame(self._ret, columns=["feature", "iv", "ks", "bins"])
        print("=======================================")
        print("IV and k-s scores of selected features:")
        print(self.IV_table)
        print("=======================================")
        if self._zerobin: print("The bins contain zero class")
        
sb = SplitBins()
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = ['a','b','c']
sb.fit(x,y)
re = sb.transform(x)
print(re)