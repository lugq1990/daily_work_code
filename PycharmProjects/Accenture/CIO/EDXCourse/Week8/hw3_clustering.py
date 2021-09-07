# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import sys
import random

# X = np.genfromtxt(sys.argv[1], delimiter=",")


def KMeans(data):
    # perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    k = 5
    n = 10
    # random sample some data as center
    # center_list = random.sample(np.arange(len(data)).tolist(), k)
    # centers = np.array([data[m] for m in center_list])

    centers = []

    random_center = data[np.random.randint(len(data))]

    data = np.array(data)

    out = []
    for i in range(n):
        belongs_list = []

        # Loop for each data point to get which cluster it belongs to
        # and with each random centers
        for d in data:
            belongs_list.append(np.argmin([np.linalg.norm(d - x) for x in centers]))

        # recompute the new center
        centerslist = []
        for j in range(k):
            centerslist.append(np.mean(data[np.array(belongs_list) == j], axis=0))


        filename = "centroids-" + str(i + 1) + ".csv"  # "i" would be each iteration
        np.savetxt(filename, centerslist, delimiter=",")
        # reassign the centers
        centers = centerslist
        out.append(centerslist)
    return out


def EMGMM(data):
    iter = 10
    k = 5
    pi = np.ones([k]) / k
    mu = np.ones([k]) / k
    sigma = np.identity(k)
    for i in range(iter):

        filename = "pi-" + str(i + 1) + ".csv"
        np.savetxt(filename, pi, delimiter=",")
        filename = "mu-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration



        for j in range(k):  # k is the number of clusters
            filename = "Sigma-" + str(j + 1) + "-" + str(i + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, _  = load_iris(return_X_y=True)
    x = x[:, :2]
    out = KMeans(x)

    print(pd.DataFrame(out))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(out)))

    plt.scatter(x[:, 0], x[:, 1])
    for c, color in zip(out, colors):
        # plt.scatter(x[:, 0], x[:, 1])
        plt.scatter(c[0][0], c[0][1], marker='o', color=color)
        plt.scatter(c[1][0], c[1][1], marker='^', color=color)
        plt.scatter(c[2][0], c[2][1], marker='v', color=color)
        plt.scatter(c[3][0], c[3][1], marker='>', color=color)
        plt.scatter(c[4][0], c[4][1], marker='<', color=color)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = np.genfromtxt(sys.argv[1], delimiter = ",")

    KMeans(data)

    EMGMM(data)
