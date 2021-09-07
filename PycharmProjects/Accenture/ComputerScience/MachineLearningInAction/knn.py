# -*- coding:utf-8 -*-
"""
This is implement with KNN algorithm

@author: Guangqiang.lu
"""
import operator
import numpy as np
import os
import matplotlib.pyplot as plt


path = "C:/Users/guangqiiang.lu/Documents/lugq/github/machinelearninginaction-master"
sub_path = "Ch02"
path = os.path.join(path, sub_path)


def create_data():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['a', 'a', 'b', 'b']
    return group, labels


def knn(input_x, dataset, labels, k):
    """compute distance with input_x and label"""
    n = dataset.shape[1]
    diss = np.linalg.norm((dataset - input_x)**2, axis=1)
    diss_index = diss.argsort()   # ascending
    class_count = {}
    for i in range(k):
        indival_label = labels[diss_index[i]]
        class_count[indival_label] = class_count.get(indival_label, 0) + 1

    sorted_class = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]


def read_file(file_name="datingTestSet.txt"):
    try:
        with open(os.path.join(path, file_name), 'r') as f:
            data = f.readlines()
    except:
        pass

    input_length = len(data)
    data_matrix = np.zeros((input_length, 2))
    label_list = []
    for i, data_line in enumerate(data):
        data_line = data_line.strip().split("\t")
        data_matrix[i, :] = data_line[:2]
        label_list.append(data_line[-1])
    return data_matrix, label_list


def plot_read_data(dataset, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if labels is not None:
        # convert label to int
        uni_dic = dict()
        for i, x in enumerate(set(labels)):
            uni_dic[x] = i
        labels = np.array([uni_dic[x] for x in labels])

    colors = ['r', 'b', 'g']
    for i, c in enumerate(colors):
        ax.scatter(dataset[labels == i, 0], dataset[labels == i, 1], c=c)

    # ax.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("data points")
    plt.show()


if __name__ == "__main__":
    dataset, labels = create_data()
    input_x = np.array([2., 1.])
    pred = knn(input_x, dataset, labels, k=2)
    print("Get prediction:", pred)

    dataset_new, labels_new = read_file()

    pred_user = knn(dataset_new[0, :], dataset_new[1:, :], labels_new[1:], k=3)
    print("user prediction:", pred_user)
    plot_read_data(dataset_new, labels_new)
