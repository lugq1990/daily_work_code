# -*- coding:utf-8 -*-
"""This class is used to implement the KNN algorithm with some sample data with different distance"""
from sklearn.datasets import load_iris
import numpy as np
from collections import Counter, defaultdict

x, y = load_iris(return_X_y=True)
x = x[:, :]

class KNN(object):
    def __init__(self, n_neighbors=4, p=2):
        """
        according to different number of neighbors to judge which classes this data point belongs to.
        For KNN model, there isn't fit step, just predict step
        :param n_neighbors: how many data points are selected for the predict
        :param p: which distance is selected for measurement
        """
        self.n_neighbors = n_neighbors
        self.p = p

    @staticmethod
    def distance(x1, x2, p=2):
        # dis_sum = np.power(np.sum(np.abs(x1 - x2), axis=1), p)
        # dis = np.power(dis_sum, 1/p)

        # I could just get the distance with numpy norm
        return np.linalg.norm((np.abs(x1 - x2)), p, axis=1)

    def fit(self, x, y):
        self.data = x
        self.label = y

    def predict(self, x_test):
        if len(self.data) == 0:
            raise RuntimeError("Model haven't been fitted, fit model first!")

        # compute the distance between the test data and train data
        # loop for ever test data
        pred_list = []
        for x in x_test:
            dis = self.distance(x, self.data)
            # sort the distance to get the first k dis
            # in fact, here shouldn't use the dictionary, cause dictionary shouldn't have the same key,
            # but for distance, there should be.
            # Here I just use the np.array to make the prediction
            pred_array = np.concatenate((dis[:, np.newaxis], self.label[:, np.newaxis]), axis=1)

            # here should get the most smaller distance data label
            sorted_array = np.array(sorted(pred_array, key=lambda x: x[0], reverse=False)[:self.n_neighbors])

            label_count = Counter(sorted_array[:, 1])
            pred_list.append(label_count.most_common()[0][0])
            # dis_label = dict(zip(dis, self.label))
            # print(dis_label)
            # dis_label = sorted(dis_label, key=lambda x: x[0], reverse=True)[:self.n_neighbors].values()
            # pred_label = Counter(dis_label).most_common()[0][0]
            # pred_list.append(pred_label)
        return np.array(pred_list)

    def score(self, x, y):
        pred = self.predict(x)
        y = np.asarray(y)
        return np.sum(y == pred) / len(y)

if __name__ == '__main__':
    model = KNN()
    model.fit(x, y)
    print(model.predict(x))
    print(model.score(x, y))

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=4)
    print(knn.fit(x, y).score(x, y))