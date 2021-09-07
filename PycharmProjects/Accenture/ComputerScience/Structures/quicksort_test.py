# -*- coding:utf-8 -*-
"""this is to implement the quicksort algorithm that to sort list,
quick sort is first to select one value from the list, compare this value
with others, then the list will be split into 2 parts, lower and larger list,
then iterate the step"""

def quicksort(arr):
    if len(arr) == 0:
        return arr
    else:
        return quicksort([x for x in arr[1:] if x < arr[0]]) \
               + [arr[0]] + \
               quicksort([x for x in arr[1:] if x > arr[0]])


a = [1, 3, 1, 3, 2, 4, 10, 2, 4, 6, 48]
print(quicksort(a))
