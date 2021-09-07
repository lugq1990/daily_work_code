from __future__ import division
import numpy as np
import sys
from collections import Counter

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")


## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):
    # this function returns the required output
    n_s = y_train.shape[0]
    y_di = dict(Counter(y_train.tolist()))
    for key, value in y_di.items():
        y_di.update({key: value/ n_s})
    y_u = list(sorted(y_di.keys()))
    m_di = dict()
    for l in y_u:
        m_di[l] = np.mean(X_train[y_train == l])
    s_di = dict()
    for l in y_u:
        s_di[l] = np.mean(np.dot((X_train[y_train == l] - m_di[l]), (X_train[y_train == l] - m_di[l]).T))

    proba_list = []
    for d in X_test:
        tmp = []
        for l in y_u:
            tmp.append(y_di[l] * (1 / np.sqrt(np.abs(s_di[l]))) * np.exp(-0.5 * np.dot(np.dot((d - m_di[l]).T, (1 / s_di[l])), (d - m_di[l]))))
        proba_list.append(tmp)
    return np.array(proba_list)


final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file

# if __name__ == '__main__':
#     from sklearn.datasets import load_iris
#     x, y = load_iris(return_X_y=True)
#     print(pluginClassifier(x, y, x))