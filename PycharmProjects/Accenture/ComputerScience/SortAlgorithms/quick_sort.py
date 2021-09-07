# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""


class QuickSort:
    def sort(self, data):
        data = self._quick_sort(data, 0, len(data) - 1)
        return data

    def _quick_sort(self, data, start, end):
        if start > end:
            return
        pivot, data = self.partition2(data, start, end)
        self._quick_sort(data, start, pivot - 1)
        self._quick_sort(data, pivot + 1, end)

        return data

    def partition(self, data, start, end):
        pivot = data[start]
        left = start
        right = end

        while left != right:
            while left < right and data[right] > pivot:
                right -= 1

            while left < right and data[left] <= pivot:
                left += 1

            if left < right:
                data[left], data[right] = data[right], data[left]

        data[left], data[start] = data[start], data[left]

        return left, data

    def partition2(self, data, start, end):
        pivot = data[start]
        mark = start

        for i in range(start + 1, end + 1):
            if data[i] < pivot:
                mark += 1
                data[mark], data[i] = data[i], data[mark]

        data[mark], data[start] = data[start], data[mark]
        return mark, data

    def stack_sort(self, data):
        stack = []
        params = {}
        start, end = 0, len(data)
        params['start'] = start
        params['end'] = end
        stack.append(params)

        while stack:
            param = stack.pop()
            pivot, data = self.partition(data, param.get('start'), param.get('end') - 1)
            print(params)
            if param.get('start') < pivot - 1:
                left_param = {}
                left_param['start'] = params.get('start')
                left_param['end'] = pivot - 1
                stack.append(left_param)
            if pivot + 1 < params.get('end'):
                right_param = {}
                right_param['start'] = pivot + 1
                right_param['end'] = params.get('end')
                stack.append(right_param)
        return data


if __name__ == '__main__':
    data = [123, 3, 12, 32, 0]
    sort = QuickSort()
    # print(sort.sort(data))
    print(sort.stack_sort(data))