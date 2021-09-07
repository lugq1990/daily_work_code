import  numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import leastsq

style.use('ggplot')

# construct dataset
def cos(x):
    return np.cos(2 * np.pi * x)

x = np.linspace(0, 1, 20)
y = [m + np.random.normal(0, .1) for m in cos(x)]
x_points = np.linspace(0, 1, 1000)
y_points = cos(x_points)

# poly function
def poly(x, weights):
    return np.poly1d(weights)(x)

# residual function
def residual(weights, x, y, kinds='l2', alpha=.002):
    res = np.abs(poly(x, weights) - y)
    if kinds == 'l2':
        res += alpha * np.square(np.linalg.norm(weights))
    if kinds == 'l1':
        res += alpha * np.abs(np.linalg.norm(weights))
    return res

# fitting
def fitting(p):
    # init weight
    weights_init = np.random.rand(p+1)
    # compute weights
    weights = leastsq(residual, weights_init, args=(x, y))[0]

    # plot
    plt.plot(x_points, y_points, label='true data')
    plt.plot(x, poly(x, weights), label='fitting line')
    plt.plot(x, y, 'bo', label='getting data')
    plt.legend()
    plt.show()

fitting(10)

