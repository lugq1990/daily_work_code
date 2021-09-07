# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
from statistics import mean

style.use('fivethirtyeight')

# first create dataset
def create_data(hm, variance, step=2, has_v=False):
    val = 1
    y = []
    for _ in range(hm):
        y.append(val+ random.randrange(-variance, variance))
        if has_v and has_v == 'pos':
            val += step
        elif has_v and has_v == 'neg':
            val -= step
    x = np.arange(1, len(y)+1)
    return np.array(x), np.array(y)

# compute the best m and b
def best_m_b(x, y):
    up = mean(x) * mean(y) - mean(x * y)
    down = mean(x) ** 2 - mean(x ** 2)
    m = up / down
    b = mean(y) - m* mean(x)
    return m, b

# compute the linear model
def linear(x, m, b):
    return x * m + b

# compute the square loss
def square(y, pred):
    return sum((y - pred) ** 2)

# compute the r2 square
def r_square(x, y, m, b):
    return 1. - square(y, linear(x, m, b)) / square(mean(y), linear(x, m, b))

def plot_re(x, y):
    plt.scatter(x, y)
    plt.plot(linear(x, m, b))
    plt.legend()
    plt.show()

x, y = create_data(80, 10, 2, 'pos')
m, b = best_m_b(x, y)
print('m={0:.4f}, b={1:.4f}'.format(m, b))
print('Total loss={0:.2f}'.format(square(y, linear(x, m, b))))
print('R2 square = {0:.2f}'.format(r_square(x, y, m, b)))
plot_re(x, y)
