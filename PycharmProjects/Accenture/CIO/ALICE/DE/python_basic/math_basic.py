# -*- coding:utf-8 -*-
"""this is just used for explaining the basic math """
import numpy as np

a = [1, 2, 3]

# 平均值：mean, (1+2+3)/3
mean = np.mean(a)

# 方差: std(先求平均值,是2)每个对象减去平均值的平方，再求和，再除以多少个对象
# std**2 = [(1 -2)**2 + (2 - 2)** 2 + (3 - 2)**2] / 3
# numpy.std为标准差，标准差**2 = 方差
std = np.std(a) ** 2

std_manually = np.sum((a - mean)**2) / len(a)

print(std == std_manually)

from scipy.optimize import minimize
