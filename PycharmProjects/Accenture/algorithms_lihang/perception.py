# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x, y = iris.data, iris.target
x, y = x[:100, 1:3], y[:100]

# cause perception label should be 1 or -1
y = [1 if m == 1 else -1 for m in y]

# update weights and bias: w = w+lr*xi*yi; b = b + yi
class Perception(object):
    def __init__(self, lr=.01, iters=10000):
        self.lr = lr
        self.iters = iters

    # this sign function
    def sign(self, x, w, b):
        return np.dot(x, w) + b

    # init w and b
    def init(self, dim):
        # self.w = np.random.random(dim)
        # self.b = np.random.random()
        self.w = np.ones(dim, dtype=np.float32)
        self.b = 0

    # this is to fit
    def fit(self, x, y):
        self.init(x.shape[1])
        has_wrong = True
        step = 0
        while has_wrong:
            wrong_num = 0
            for i in range(len(x)):
                if y[i] * self.sign(x[i], self.w, self.b) < 0:
                    self.w = self.w + self.lr * x[i] * y[i]
                    self.b = self.b + self.lr * y[i]
                    wrong_num += 1
                step += 1
            if wrong_num == 0 or step > self.iters:
                print('Final step:', step)
                has_wrong = False

    def predict(self, x):
        pred = np.dot(x, self.w) + self.b
        pred = [1 if m >= .5 else 0 for m in pred]
        return np.array(pred)

    def score(self, x, y):
        pred = self.predict(x)
        y = np.asarray(y)
        return np.sum(pred == y) / len(y)

    # here to plot the original dataset and prediction and decision function line
    def plot(self, x, y):
        pred = self.predict(x)

        y = np.asarray([1 if m == 1 else 0 for m in y])

        # here to plot the decision function
        x_points = np.linspace(min(x[:, 0]), max(x[:, 0]), len(x))
        # y_points = x_points * self.w[0] + self.b
        # y_points = -(self.w[0] * x_points + self.b) / self.w[1]
        y_points = np.dot(x, self.w) + self.b

        plt.plot(x_points, y_points, label='decision function')
        # loop for the datapoints
        color_list = ['green', 'black']

        for i in range(len(x)):
            if y[i] != pred[i]:
                plt.scatter(x[i, 0], x[i, 1], color='red')
            else:
                plt.scatter(x[i, 0], x[i, 1], color=color_list[y[i]])
        plt.show()


model = Perception()
model.fit(x, y)

y_true = np.array([1 if m == 1 else 0 for m in y])
pred = model.predict(x)
print(model.score(x, y_true))
model.plot(x, y)


from sklearn.linear_model import Perceptron
