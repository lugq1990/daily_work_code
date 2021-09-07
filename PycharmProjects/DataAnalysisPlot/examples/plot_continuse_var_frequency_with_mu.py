# -*- coding:utf-8 -*-
import pandas as pd
from analysisplot.univariate import plot_continuous_var_frequency_with_mu

path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')


plot_continuous_var_frequency_with_mu(df,)