# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_over_ratio_curve(data, range=10):
    score_s = data['all_score']
    min_s = score_s.min()
    max_s = score_s.max()
    split_r = int((max_s - min_s)/range) + 1
    ratio_dict = dict()
    f = lambda x: True if x >= min_s + i * range and x < min_s + (i + 1) * range else False
    for i in arange(split_r):
        is_sati = data['all_score'].apply(f)
        sati_d = data[is_sati]
        over_d = sati_d[sati_d['over_dues'] != 0]
        ratio_dict[((min_s+i*range)+(min_s+(i+1)*range))/2] = (over_d.shape[0]/sati_d.shape[0])
    plt.plot(ratio_dict.values())
    plt.xticks(np.arange(len(ratio_dict)),list(ratio_dict.keys()))
    plt.title('不同授信分数段逾期率曲线')
    plt.show()
    return ratio_dict



