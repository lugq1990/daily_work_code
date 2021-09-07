# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:59:57 2017

@author: Administrator
"""

import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import os
import time

from PyML.feature_selection import Binning




CURRENT_DIR = os.path.dirname(__file__)
FILE_DIR = os.path.join(CURRENT_DIR, "drawn_data.csv")
WRITE_DIR = os.path.join(CURRENT_DIR, "IV_table.csv")
DATA_DIR = os.path.join(CURRENT_DIR, "data_woe.csv")
        
start = time.time()
binning = Binning.SplitBins(max_iter=30, verbose=True, duplicates='drop', zerobin=True, woe=True)

dat = pd.read_csv(FILE_DIR, encoding="gb18030")

col_names = dat.columns.tolist()
col_names.remove("uid")
col_names.remove("label")
X = dat[col_names]
y = dat["label"]
splited_table = binning.fit_transform(X, y)




#print splited_table
#splited_table.to_csv(DATA_DIR, encoding="gb18030", index=False)
#binning.IV_table.to_csv(WRITE_DIR, encoding="gb18030", index=False)
print(time.time() - start)
