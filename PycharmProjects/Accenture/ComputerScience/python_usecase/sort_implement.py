# -*- coding:utf-8 -*-
"""
This is to implement different sort functionality

@author: Guangqiang.lu
"""


def bubble_sort(array):
    for i in range(len(array)):
        for j in range(len(array) - 1):
            if array[j] > array[j + 1]:
                tmp = array[j + 1]
                array[j + 1] = array[j]
                array[j] = tmp

    return array


def select_sort(array):
    for i in range(len(array)):
        min_index = 0
        for j in range(i + 1, len(array)):
            if array[j] < array[min_index]:
                min_index = j

        array = change(array, i, min_index)
        return array


def reverse_array(array):
    for i in range(int(len(array) / 2)):
        array = change(array, i, len(array) - 1 - i)
    return array


def change(array, i, index):
    if i != index:
        array[i], array[index] = array[index], array[i]
    return array


if __name__ == '__main__':
    data = [1, 2, 32, 4, 6, 18, 49]
    print(bubble_sort(data))
    print(select_sort(data))
    print(reverse_array(select_sort(data)))