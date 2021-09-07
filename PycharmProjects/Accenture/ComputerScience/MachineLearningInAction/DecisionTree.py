# -*- coding:utf-8 -*-
"""
Decision tree implement

@author: Guangqiang.lu
"""
from math import log
import numpy as np
import operator


def create_dataset():
    # array data type must have same data type!
    # if we face with any string, then will convert to string.
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels


def compute_ent(data_list):
    """entropy is computed based on the probability, with sum(-log2(pro))
    but here just based on the last column"""
    feature_dic = dict()
    for d in data_list:
        d = d[-1]
        feature_dic[d] = feature_dic.get(d, 0) + 1
    feature_dic = {k: -log(v / len(data_list), 2) for k, v in feature_dic.items()}
    return sum(feature_dic.values())


def split_dataset(dataset, axis, value):
    res_data = []
    for f in dataset:
        if f[axis] == value:
            sati_data = f[:axis]
            sati_data.extend(f[axis+1: ])
            res_data.append(sati_data)
    return res_data


def choose_best_feature_to_split(dataset):
    """this is try to get the best feature index to split data"""
    n_features = len(dataset[0]) - 1
    base_ent = compute_ent(dataset)
    best_info_gain = 0.
    best_feature = -1
    for i in range(n_features):
        feature_list = [x[i] for x in dataset]
        uni_feature = set(feature_list)
        new_ent = 0.
        # loop for each value to split data and compute entropy
        for value in uni_feature:
            sub_dataset = split_dataset(dataset, i, value)
            # we get sub dataset
            sub_prob = len(sub_dataset) / len(dataset)
            new_ent += sub_prob * compute_ent(sub_dataset)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def vote(class_list):
    """this is to get the most frequent class as label"""
    class_dic = {}
    for cl in class_list:
        class_dic[cl] = class_dic.get(cl, 0) + 1

    sorted_class = sorted(class_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]


def create_tree(dataset, labels):
    class_list = [x[-1] for x in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        # same number with class
        return class_list[0]
    if len(class_list[0]) == 1:
        # get the most frequent
        return vote(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    mytree = {best_feature_label: {}}
    del(labels[best_feature])
    # get the best feature value
    feature_value = [x[best_feature] for x in dataset]
    uni_feature_value = set(feature_value)
    for value in uni_feature_value:
        sub_labels = labels[:]
        # recursive to create tree
        mytree[best_feature_label][value] = create_tree(
            split_dataset(dataset, best_feature, value), sub_labels)

    return mytree


def classifier(input_tree, feature_labels, test_vec):
    root_str = input_tree.keys()[0]
    second_dict = input_tree[root_str]
    feature_index = feature_labels.index(root_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classifier(second_dict[key], feature_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == "__main__":
    dataset, labels = create_dataset()
    print(compute_ent(dataset))
    print(split_dataset(dataset, 0, 1))
    print(choose_best_feature_to_split(dataset))
    # print(create_tree(dataset, labels))
    my_tree = create_tree(dataset, labels)
    # print(classifier(my_tree, labels, [1, 1]))
