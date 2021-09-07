
def bubble_sort(data):
    for i in range(len(data)):
        for j in range(1, len(data)):
            if data[j] < data[j -1]:
                data[j], data[j - 1] = data[j - 1], data[j]
    
    return data


def select_sort(data):
    for i in range(len(data)):
        min_index = i
        for j in range(i, len(data)):
            if data[j] < data[min_index]:
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]

    return data


def insert_sort(data):
    for i in range(1, len(data)):
        for j in range(i):
            if data[j] > data[i]:
                data[i], data[j] = data[j], data[i]
    
    return data


def merge_sort(data):
    if len(data) < 2:
        return data
    mid_index = int(len(data) / 2)
    left, right = data[:mid_index], data[mid_index:]

    return _merge(left, right)


def _merge(left, right):
    res = []
    while left and right:
        if left[0] < right[0]:
            res.append(left.pop(0))
        else:
            res.append(right.pop(0))
    
    while left:
        res.append(left.pop(0))
    while right:
        res.append(right.pop(0))

    return res


def quick_sort(data):
    if not data:
        return
    _quick_sort(data, 0, len(data) - 1)
    return data

def _quick_sort(data, left, right):
    if left < right:
        partition_key = _get_partition_key_v2(data, left, right)
        _quick_sort(data, left, partition_key- 1)
        _quick_sort(data, partition_key+1, right)


def _get_partition_key_v2(data, left, right):
    partition_value = data[left]
    mark = left

    for i in range(left + 1, right+1):
        if data[i] < partition_value:
            mark += 1
            data[i], data[mark] = data[mark], data[i]
    
    data[left], data[mark] =data[mark], data[left]
    return mark

def _get_partition_key(data, left, right):
    partition_value = data[left]
    
    while left < right:
        while left < right and data[right] >= partition_value:
            right -= 1
        data[right] = data[left]
        while left < right and data[left] <= partition_value:
            left +=1
        data[left] = data[right]
    data[left] = partition_value
    return left


def quick_sort_stack(data):
    param_stack = []
    param_dict = {}
    param_dict['start'] = 0
    param_dict['end'] = len(data) - 1
    param_stack.append(param_dict)

    while param_stack:
        param_dict = param_stack.pop()
        partition_key = _get_partition_key(data, param_dict['start'], param_dict['end'])
        if partition_key < param_dict['start']:
            param_dict['start'] = param_dict.get('start')
            param_dict['end'] = partition_key - 1
            param_stack.append(param_dict)
        if partition_key +1 < param_dict['end']:
            param_dict['start'] = partition_key + 1
            param_dict['end'] = param_dict.get('end')
            param_stack.append(param_dict)
    return data



def binary_search(data, value):
    index = _binary_search(data, value, 0, len(data) - 1)

    return index

def _binary_search(data, value, left, right):
    if left <= right:
        mid_index = left + int((right - left)/2)

        if data[mid_index] == value:
            return mid_index
        elif data[mid_index] < value:
            return _binary_search(data, value, mid_index+1, right)
        else:
            return _binary_search(data, value, left, mid_index-1)
    else:
        return -1

    
def count_sort(data):
    import numpy as np
    max_value = 0
    for x in data:
        if x > max_value:
            max_value = x
    
    tmp_array = np.zeros(shape=(max_value+1, ), dtype=np.int)

    for i in range(len(data)):
        tmp_array[data[i]] += 1
    print(tmp_array)
    res = []
    for i in range(len(tmp_array)):
        if tmp_array[i] == 0:
            continue
        for _ in range(tmp_array[i]):
            res.append(i)
    
    return res


def bucket_sort(data):
    min_value, max_value = 0, 0
    for i in range(len(data)):
        if data[i] < min_value:
            min_value = data[i]
        if data[i] > max_value:
            max_value = data[i]
    
    diff = max_value - min_value

    bucket_list = []
    for i in range(len(data)):
        bucket_list.append([])
    
    for i in range(len(data)):
        bucket_index = int((data[i] - min_value) * (len(data) - 1)/ diff)
        bucket_list[bucket_index].append(data[i])
    
    for i in range(len(data)):
        bucket_list[i] = sorted(bucket_list[i])

    print(bucket_list)
    res = []
    for bucket in bucket_list:
        for x in bucket:
            res.append(x)
    
    return res


 
if __name__ == '__main__':
    data = [1,3, 2, 5, 8, 6, 6, 7, 10]

    print(bubble_sort(data))

    print(select_sort(data))

    print(insert_sort(data))

    print(merge_sort(data))

    print("quick:", quick_sort(data))

    print(quick_sort_stack(data))

    sorted_data = quick_sort(data)
    print(binary_search(sorted_data, 10))

    print(count_sort(data))

    print(bucket_sort(data))