# -*- coding:utf-8 -*-
"""
This is the most basic implement with different search
algorithms.

@author: Guangqiang.lu
"""


class BinarySearch:
    def search(self, data, key):
        """
        Get the index of the key value from array
        Parameters
        ----------
        data list type
        key value
        Returns
        -------
        index of key or -1 means doesn't find it.
        """
        start_index = 0
        end_index = len(data) - 1
        while start_index <= end_index:
            # we should start with start index plus the mid value
            mid = start_index + int((end_index - start_index) / 2)
            if key == data[mid]:
                return mid
            elif key < data[mid]:
                end_index = mid - 1
            else:
                start_index = mid + 1
        return -1

    def contains(self, data,  key):
        return self.search(data, key) != -1


if __name__ == '__main__':
    binary = BinarySearch()
    data = [1, 2, 12, 32, 432]
    print(binary.search(data, 3))
    print(binary.contains(data, 1))
