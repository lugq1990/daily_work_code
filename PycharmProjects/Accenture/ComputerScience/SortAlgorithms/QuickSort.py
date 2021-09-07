# -*- coding:utf-8 -*-
"""
This is python implement with quick sort, most powerful
sort algorithm with time: n*log(n)

@author: Guangqiang.lu
"""
import random


class Sort(object):
    def sort(self):
        pass

    def less(self, x, y):
        return x < y

    def exch(self, i, j):
        tmp = self.data[i]
        self.data[i] = self.data[j]
        self.data[j] = tmp

    def show(self):
        print("Now data is: ")
        print(self.data)

    def is_sort(self):
        for i in range(1, len(self.data)):
            if self.less(self.data[i], self.data[i - 1]):
                raise Exception("Not sort!")
        return True


class QuickSort(Sort):
    def __init__(self):
        super(QuickSort, self).__init__()

    def sort(self, data):
        self.data = data
        random.shuffle(self.data)
        self.sort_part(0, len(self.data) - 1)
        return self.data

    def sort_part(self, low, high):
        if low >= high:
            return
        j = self.partition(low, high)
        # left part to sort
        self.sort_part(low, j - 1)
        # right part to sort
        self.sort_part(j + 1, high)

    def partition(self, low, high):
        """this is main logic to get the best
        split index and the sorted data with
        that index left is smaller and right
        is greater."""
        value = self.data[low]
        i = low + 1
        j = high
        while True:
            while self.less(self.data[i], value):
                i += 1
                if i >= high:
                    break
            while self.less(value, self.data[j]):
                j -= 1
                if j <= low:
                    break
            if i >= j:
                break
            self.exch(i, j)
        # we are free
        self.exch(low, j)
        return j


class SelectionSort(Sort):
    """For select sort time is n**2"""
    def __init__(self):
        super(SelectionSort, self).__init__()

    def sort(self, data):
        self.data = data
        for i in range(len(data)):
            min_index = i
            for j in range(i, len(data)):
                if self.less(self.data[j], self.data[min_index]):
                    min_index = j
            self.exch(i, min_index)
        return self.data


if __name__ == "__main__":
    data = [4, 2, 5, 7, 20]
    data_select = [4, 2, 5, 7, 20]
    print("Quick Sort")
    quick_sort = QuickSort()
    quick_sort.sort(data)
    quick_sort.show()

    print("Selection Sort")
    select_sort = SelectionSort()
    select_sort.sort(data_select)
    select_sort.show()

