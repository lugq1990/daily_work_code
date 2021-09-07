# first let's try with bubble sort
from abc import ABCMeta, abstractmethod


class Sort(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sort(self, data):
        raise NotImplemented


class BubbleSort(Sort):
    def sort(self, data):
        for _ in range(len(data)):
            for i in range(1, len(data)):
                if data[i] < data[i - 1]:
                    data[i], data[i - 1] = data[i - 1], data[i]

        return data


class SelectSort(Sort):
    def sort(self, data):
        for i in range(len(data)):
            min_index = i
            for j in range(i, len(data)):
                if data[j] < data[min_index]:
                    min_index = j
            data[i], data[min_index] = data[min_index], data[i]
    
        return data


class InsertSort(Sort):
    def sort(self, data):
        for i in range(1, len(data)):
            for j in range(i):
                if data[j] > data[i]:
                    data[j], data[i] = data[i], data[j]
        return data


class MergeSort(Sort):
    def sort(self, data):
        if len(data) == 1:
            return data
        mid_index = int(len(data) / 2)
        left_data = data[:mid_index]
        right_data = data[mid_index:]
        return self.sort(left_data) + self.sort(right_data) 


class QuickSort(Sort):
    def sort(self, data):
        self._sort(data, 0, len(data) - 1)

        return data

    def _sort(self, data, left, right):
        if left < right:
            partition_key = self.get_partition_key(data, left, right)
            self._sort(data, left, partition_key - 1)
            self._sort(data, partition_key+1, right)

    def get_partition_key(self, data, left, right):
        partition_value = data[left]
        
        while left < right:
            while left < right and data[right] >= partition_value:
                right -= 1
            data[right] = data[left]
            while left < right and data[left] <= partition_value:
                left += 1
            data[left] = data[right]
        data[left] = partition_value
        return left




if __name__ == '__main__':
    data = [3, 1, 5, 10, 7, 8, 11, 4]

    bubble_sort = BubbleSort()
    print("Bubble: ", bubble_sort.sort(data))

    select_sort = SelectSort()
    print("Select: ", select_sort.sort(data))

    insert_sort = InsertSort()
    print("Insert: ", insert_sort.sort(data))

    merge_sort = MergeSort()
    print("Merge: ", merge_sort.sort(data))

    quick_sort = QuickSort()
    print("Quick: ", quick_sort.sort(data))
