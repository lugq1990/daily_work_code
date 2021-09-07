# -*- coding:utf-8 -*-
"""
This is to implement the basic logic with map reduce
to count the word appears times. As we have to define
the parent class for sub-class to inherent, like map
function to read data, reduce function to do the aggregation
function.

@author: Guangqiang.lu
"""
import tempfile
import os
from threading import Thread


tmp_path = tempfile.mkdtemp()


def create_sample_data(n=3):
    for i in range(n):
        with open(os.path.join(tmp_path, 'test_{}.txt'.format(str(i))), 'w') as f:
            f.write("1" * i * 100)


# first should input class
class InputData:
    def read(self):
        raise NotImplementedError


class LineInputData(InputData):
    def __init__(self, path):
        super(LineInputData, self).__init__()
        self.path = path

    def read(self):
        return open(self.path).read()


class Worker:
    def __init__(self, data):
        self.data = data
        self.res = None

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError


class LineWorker(Worker):
    def map(self):
        data = self.data.read()
        self.res = data.count('1')

    def reduce(self, other):
        self.res += other.res


# then we should combine whole input data and worker
def generate_inputs(data_dir):
    for file in os.listdir(data_dir):
        yield LineInputData(os.path.join(data_dir, file))


# then we have to create many workers
def create_workers(input_list):
    workers = []
    for input_data in input_list:
        workers.append(LineWorker(input_data))
    return workers


# then should use whole thing to execute the thread
def execute(workers):
    threads = [Thread(target=w.map) for w in workers]
    for t in threads: t.start()
    for t in threads: t.join()

    first_worker, others = workers[0], workers[1:]
    for w in others:
        first_worker.reduce(w)
    return first_worker.res


def mapreduce(data_dir):
    input_list = generate_inputs(data_dir)
    workers = create_workers(input_list)
    return execute(workers)


if __name__ == '__main__':
    create_sample_data()
    print("Get: ", mapreduce(tmp_path))

