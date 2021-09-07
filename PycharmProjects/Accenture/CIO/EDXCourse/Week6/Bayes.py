# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import sys

# X_train = np.genfromtxt(sys.argv[1], delimiter=",")
# y_train = np.genfromtxt(sys.argv[2])
# X_test = np.genfromtxt(sys.argv[3], delimiter=",")


## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):
    # this function returns the required output

    # First step to compute class priors
    y_train = y_train.reshape(-1)
    n_samples = y_train.shape[0]
    from collections import Counter
    pi_dic = dict(Counter(y_train.tolist()))
    for k, v in pi_dic.items():
        pi_dic.update({k: v / n_samples})

    # Get unique labels
    y_unique = list(sorted(pi_dic.keys()))
    # Second step is to compute the class conditional density, according to MLE
    # 1.Compute mu
    mu_dict = dict()
    for label in y_unique:
        mu_dict[label] = np.mean(X_train[y_train == label])
    # 2. Compute sigma
    sigma_dict = dict()
    for label in y_unique:
        # sigma_dict2[label] = np.mean(np.dot((X_train[y_train == label] - mu_dict[label]), (X_train[y_train == label] - mu_dict[label]).T))
        n_y = len(X_train[y_train == label])
        sigma_dict[label] = np.cov(X_train[y_train == label]) / n_y

    print(sigma_dict)

    # Return predicted result
    prob_list = []
    for data in X_test:
        prob_x = []
        for la in y_unique:
            # score = pi_dic[la] * np.abs(sigma_dict[la])**(-.5) * np.exp(-.5 * np.dot(np.dot((data - mu_dict[la]).T, sigma_dict[la]**(-1)), (data - mu_dict[la])))
            score = pi_dic[la] * (1 / np.sqrt(np.abs(sigma_dict[la]))) * np.exp(-.5 * np.dot(np.dot((data - mu_dict[la]).T, (1/sigma_dict[la])), (data - mu_dict[la])))
            prob_x.append(score)
        prob_x = prob_x / np.sum(prob_x)
        prob_list.append(prob_x)

    return np.array(prob_list)


# final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function
#
# np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)
    x = x[:, 0]
    re = pluginClassifier(x, y, x)
    pred = np.argmax(re, axis=1)
    # for p, t in zip(pred, y):
    #     print('True is %d, pred is %d'%(t, p))
    from sklearn.metrics import accuracy_score
    print('accuracy is: ', accuracy_score(y, pred))
