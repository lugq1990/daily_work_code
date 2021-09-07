# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import numpy as np


class CountSort:
    def sort(self, data):
        max = data[0]
        for i in range(len(data)):
            if data[i] > max:
                max = data[i]

        data_array = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            data_array[data[i]] += 1

        return data_array

    def sort_stable(self, data):
        min = data[0]
        max = data[0]
        for i in range(len(data)):
            if data[i] > max:
                max = data[i]
            if data[i] < min:
                min = data[i]

        count_array = np.zeros(max - min + 1, dtype=int)
        for i in range(len(data)):
            count_array[data[i] - min] += 1

        for i in range(1, len(count_array)):
            count_array[i] += count_array[i - 1]

        sort_array = np.zeros(len(data), dtype=int)
        for i in range(len(data) - 1, 0, -1):
            sort_array[count_array[data[i] - min] - 1] = data[i]
            count_array[data[i] - min] -= 1

        return sort_array


class BucketSort:
    def sort(self, data, bucket_num=None):
        min = data[0]
        max = data[0]
        for i in range(len(data)):
            if data[i] > max:
                max = data[i]
            if data[i] < min:
                min = data[i]

        diff = max - min
        bucket_num = len(data) if bucket_num is None else bucket_num
        bucket_dict = dict()
        for i in range(bucket_num):
            bucket_dict[i] = []

        for i in range(len(data)):
            n = int((data[i] - min) * (bucket_num - 1) / diff)
            bucket_dict.get(n).append(data[i])

        for i in range(bucket_num):
            bucket_dict[i].sort()

        res = []
        for i in range(bucket_num):
            res.extend(bucket_dict.get(i))

        return res


if __name__ == '__main__':
    data = [1, 2, 3, 2, 4, 5, 2]
    count = CountSort()
    print(count.sort(data))

    print(count.sort_stable(data))

    bucket = BucketSort()
    print(bucket.sort(data))
