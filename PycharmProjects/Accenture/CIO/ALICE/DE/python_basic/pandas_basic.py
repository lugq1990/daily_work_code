# -*- coding:utf-8 -*-
"""numpy is the basic data construct module, numpy.ndarray is a n dimension array,
array is more efficient than list, as list could stores many other type data,
but array could just store save type values.
For pandas.DataFrame is built based on numpy.ndarray, but DataFrame could just have 2 dimension,
np.ndarray could have many dimensions, pd.Series is based on 1 dimension data, could be 1 dimension
arrary or just list, as bellow:"""

import numpy as np
import pandas as pd

# create random float array
a = np.random.randn(10, 3)
print('shape of a:', a.shape)
print("data type of a:", type(a))
print("storage data type of a, means with what data type is stored:", a.dtype)

# convert np.array to list
a_list = a.tolist()
print("converted list:", a_list)

# create random int with random module
b_int = np.random.randint(low=5, high=20, size=(10, ))
print("create random int: ", b_int)
print("shape: ", b_int.shape)
print("type:", b_int.dtype)
# convert array to list
b_int_list = b_int.tolist()
print("converted:", b_int_list)

# pandas dataframe could be created from the np.ndarrry
df = pd.DataFrame(a)
# serises with list
series = pd.Series(b_int_list)
# series with 1 dimension array
series_array = pd.Series(b_int)
# Noted: series is 1 dimension, if you give it with 2D(D means dimension), will raise error
series_error = pd.Series(a)