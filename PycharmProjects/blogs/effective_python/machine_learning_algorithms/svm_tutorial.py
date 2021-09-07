import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris


x, y = load_iris(return_X_y=True)
x = x[:, :2]

clf = SVC(kernel='linear', C=1)

clf.fit(x, y)


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

h = (x_max / x_min) / 100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def plot_c(c_list, kernal_type='linear'):

    fig, ax = plt.subplots(1, len(c_list))

    for i in range(len(c_list)):
        clf.C = c_list[i]
        clf.kernel = kernal_type

        clf.fit(x, y)
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax[i].contourf(xx, yy, z, cmap=plt.cm.Paired)

        ax[i].scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

    plt.show()


c_list = [1, 10, 100, 1000]

plot_c(c_list, kernal_type='rbf')



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
