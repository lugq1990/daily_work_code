# -*- coding:utf-8 -*-
import pandas as pd
from analysisplot.univariate import plot_all_var_frequency

path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')

plot_all_var_frequency(df)
