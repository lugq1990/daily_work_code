# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 08:23:44 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import os
import time

from sklearn.cluster import k_means
from PyML.feature_selection import Binning


BASE_DIR = os.path.dirname(__file__)
DATA_DIR00 = os.path.join(BASE_DIR, "credit_score_bins.csv")
DATA_DIR = os.path.join(BASE_DIR, "data00.csv")
NDATA_DIR = os.path.join(BASE_DIR, "data04.csv")


dat = pd.read_csv(DATA_DIR)
new_dat = dat[["phone", "overdue"]]
col_names = dat.columns.tolist()
col_names.remove("phone")
col_names.remove("overdue")
col_names.remove("credit_score")

binning = Binning.SplitBins(max_iter=30, verbose=True, duplicates='drop', zerobin=True, woe=False)
new_dat['credit_score_bins'] = binning.fit_transform(dat[['credit_score']], dat["overdue"])

print new_dat
#new_dat.to_csv(DATA_DIR00, index=False)

start = time.time()
for i1, i2, i3 in zip(range(0, 54, 3),range(1, 54, 3),range(2, 54, 3)):
    name = col_names[i1].split("_")[0]
    df = dat[["overdue"]]
    X = dat[[col_names[i1], col_names[i2], col_names[i3]]]
    best_ks = 0
    best_iv = 0
    best_bins = 0
    best_splited = None
    for k in xrange(5, 30):
        for _ in range(5):
            df[name] = k_means(X, k, init='random')[1]
            tbl = df[["overdue", name]].groupby(name).count() * 1.
            tbl["yes_overdue"] = df[["overdue", name]].groupby(name).sum() * 1.
            tbl["no_overdue"] = tbl["overdue"] - tbl["yes_overdue"]
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
                best_bins = k            
                best_splited = df[name].copy()
                woe_dict = tbl["woe"].copy()
                woe_dict[woe_dict==np.inf] = 0
    new_dat[name] = best_splited.apply(lambda x: woe_dict[x])
    print name, best_iv, best_ks, best_bins
    
new_dat.to_csv(NDATA_DIR, index=False)
print time.time() - start

            
            
            