# This is used for testing with different time series problem hypothesis testing

from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

x, y = load_boston(return_X_y=True)
random = np.random.randn(100)

# 1. 白噪声检验(数据是随机数据，预测难度大)
#  首先假设数据为白噪声数据，使用假设检验，如果P值小于阈值（一般0.05），则拒绝假设，则数据有规律
# 如果数据为白噪声，则很难能使用模型来进行预测。

# 全部的假设检验都在包：statsmodels里面了
from statsmodels.stats.diagnostic import acorr_ljungbox as ljbox 

# This is to check with real data.
res = ljbox(y, lags=1)
print(res)
# return: (array([274.69471768]), array([1.07584075e-61]))
# 第2项为P值，1.07584075e-61 接近于0，所以小于P=0.05,拒绝假设：数据为白噪声。数据为真实数据，所以确实不是白噪声

res = ljbox(random,lags=1)
print(res)
# retrun: (array([0.34936306]), array([0.55447391]))
# 第2项为0.55447391，远大于P=0.05，所以接受假设，数据确实为白噪声


# Let's plot the data to see more details
fig = plt.figure(figsize=(12, 10))
plt.plot(y)
plt.title("Real data not white noise")
plt.show()

fig = plt.figure(figsize=(12, 10))
plt.plot(random)
plt.title("Random data as white noise")
plt.show()