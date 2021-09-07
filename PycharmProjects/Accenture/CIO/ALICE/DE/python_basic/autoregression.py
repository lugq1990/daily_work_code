# -*- coding:utf-8 -*-
"""This is to implement the autoregression model as AR model from statsmodels module
and compute the autocorrelation from scratch"""

import numpy as np
from statsmodels.tsa.ar_model import AR
from sklearn.datasets import load_boston
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt


# load some sample data
_, y = load_boston(return_X_y=True)
# plot_acf(y, lags=30)

# split the data to train and test
train = y[:-5]
test = y[-5:]

# build the AR model
model = AR(train)
model = model.fit()

# print model weights and lags
print('lags: ',model.k_ar)
print('model weights for different lags:',model.params)

# Here is to compute the autocorrelation with manually function
def auto_corr_k(y, k):
    return (np.sum((y[:len(y) - k])*(y[k:])) / np.sum((y - y.mean())**2))

k_list = [4, 6, 10, 50]
for k in k_list:
    print('auto corr with K=%d : %f'%(k, auto_corr_k(y, k)))

# plot the autocorelation
plt.plot(np.arange(1, 50, 5), [auto_corr_k(y, k) for k in np.arange(1, 50, 5)])
plt.title('Different auto correlation with different k lags')
plt.xlabel('k lags')
plt.ylabel('auto correlation')
plt.show()


from sklearn.preprocessing import RobustScaler