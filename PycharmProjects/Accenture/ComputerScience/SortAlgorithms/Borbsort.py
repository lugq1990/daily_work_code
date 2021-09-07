# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""


def bort_sort(arr):
    for i in range(len(arr)):
        is_sort = True
        boarder = len(arr) - 1
        for j in range(boarder):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                is_sort = False
                boarder = j
        if is_sort:
            break

    return arr


if __name__ == '__main__':
    data = [123, 3, 12, 32, 0]
    print(bort_sort(data))