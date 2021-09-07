# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""


class StampSort:
    def sort(self, data):
        self.data = data
        for i in range(int(len(data) / 2) - 1, 0, -1):
            self.down_adjust(self.data, i, len(data) - 1)

        for x in self.data:
            print(x, end=' ')

        for i in range(len(self.data) - 1, 0, -1):
            self.data[i], self.data[0] = self.data[0], self.data[i]
            self.down_adjust(self.data, 0, i)

        return self.data

    def down_adjust(self, data, parent_index, length):
        tmp = data[parent_index]
        child_index = 2 * parent_index + 1

        while child_index < length:
            if child_index + 1 < length and data[child_index + 1] > data[child_index]:
                child_index += 1

            if tmp >= data[child_index]:
                break

            data[parent_index] = data[child_index]
            parent_index = child_index
            child_index = 2 * parent_index + 1
        data[parent_index] = tmp


if __name__ == '__main__':
    import numpy as np
    data = np.array([123, 3, 12, 32, 0])
    stamp = StampSort()
    print(stamp.sort(data))
