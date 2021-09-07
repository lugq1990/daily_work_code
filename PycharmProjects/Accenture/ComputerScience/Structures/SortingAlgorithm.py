# -*- coding:utf-8 -*-
import numpy as np


"""This is an implement for bubble sort"""
def bubble_sort(arr):
    n = len(arr)
    x = -1
    fini = True

    # While loop util all finished
    while fini:
        fini = False
        x += 1
        for i in range(1, n - x):
            if arr[i-1] > arr[i]:
                arr[i-1], arr[i] = arr[i], arr[i-1]
                fini = True

    return arr

"""This is an implement for Selection sort"""
def select_sort(arr):
    for i in range(len(arr)):
        m = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[m]:
                m = j
        arr[i], arr[m] = arr[m], arr[i]

    return arr

"""This is an implement for Insert sort"""
def insert_sort(arr):
    for i in range(len(arr)):
        curr = arr[i]
        pos = i

        # While loop to change the value, because every time we can sort list apart
        # if before value is greater than current value, change it
        while pos > 0 and arr[pos - 1] > curr:
            arr[pos] = arr[pos - 1]
            pos -= 1
        arr[pos] = curr
    return arr




if __name__ == '__main__':
     a = [0, 10, 2, 9, 3, -1, -10]
     print('Bublle sort:', bubble_sort(a))
     print('Selection sort:', select_sort(a))
     print('Insert sort:', insert_sort(a))