# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import numpy as np


def get_max_diff(data_list):
    min_d = data_list[0]
    max_d = data_list[0]
    for i in range(len(data_list)):
        if data_list[i] > max_d:
            max_d = data_list[i]
        if data_list[i] < min_d:
            min_d = data_list[i]

    diff = max_d - min_d
    bucket_num = len(data_list)
    bucket_dict = {}
    for i in range(bucket_num):
        bucket_dict[i] = []

    for i in range(len(data_list)):
        num = int((data_list[i] - min_d) * (bucket_num - 1) / diff)
        bucket_dict.get(num).append(data_list[i])

    max_diff = 0
    start = 0
    max_pre = 0
    min_next = 0
    while start < bucket_num - 1:
        pre_b = bucket_dict.get(start)
        if pre_b is None:
            start += 1
            continue
        max_pre = max(pre_b)
        next_b = bucket_dict.get(start + 1)
        if next_b is None:
            start += 1
            continue
        min_next = min(next_b)
        max_diff = max_pre - min_next

    return max_diff


if __name__ == '__main__':
    data = [1, 3, 5, 10, 49, 1]
    print(get_max_diff(data))

    from sklearn.cluster import KMeans