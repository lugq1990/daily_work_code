# -*- coding:utf-8 -*-
import pandas as pd
from analysisplot.univariate import plot_continuous_vars_distribution
from analysisplot.univariate import plot_descrite_vars_frequency

path = 'E:\ExaData\plotData\wine'
df = pd.read_csv(path+'/winequality-red.csv', sep=';')

# plot_continuous_vars_distribution(df)
plot_descrite_vars_frequency(df)
