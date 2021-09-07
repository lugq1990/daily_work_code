# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import os
import shutil
from threading import Thread
import tempfile


tmp_path = tempfile.mkdtemp()
n = 10


def sample_data():
    for i in range(n):
        with open(os.path.join(tmp_path, 't_{}.txt'.format(i)), 'w') as f:
            f.write('a' * i * 10)
    print("sample finished.")


class InputData:
    def read(self):
        raise NotImplementedError


class PathInputData(InputData):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def read(self):
        return open(self.path).read()


class Worker:
    def __init__(self, input_data):
        self.input_data = input_data
        self.res = None

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError


class LineWorker(Worker):
    def map(self):
        data = self.input_data.read()
        self.res = data.count('a')

    def reduce(self, other):
        self.res += other.res


def generate_inputs(dir):
    for name in os.listdir(dir):
        yield PathInputData(os.path.join(dir, name))


def create_workers(input_list):
    workers = []
    for input_data in input_list:
        workers.append(LineWorker(input_data))
    return workers


def execute(workers):
    threads = [Thread(target=w.map) for w in workers]

    for t in threads: t.start()
    for t in threads: t.join()

    first, other = workers[0], workers[1:]
    for w in other:
        first.reduce(w)
    return first.res


def map_reduce(data_dir):
    inputs = generate_inputs(data_dir)
    workers = create_workers(inputs)
    return execute(workers)


class GeneralInputData:
    def read(self):
        raise NotImplementedError

    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError


class PathInputData(GeneralInputData):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def read(self):
        return open(self.path).read()

    @classmethod
    def generate_inputs(cls, config):
        data_dir = config['data_dir']
        for file in os.listdir(data_dir):
            yield cls(os.path.join(data_dir, file))


class GeneralWorker:
    def __init__(self, input_data):
        self.input_data = input_data
        self.res = None

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError

    @classmethod
    def create_workers(cls, input_class, config):
        workers = []
        for input_data in input_class.generate_inputs(config):
            workers.append(cls(input_data))
        return workers


class LineWorker(GeneralWorker):
    def map(self):
        data = self.input_data.read()
        self.res = data.count('a')

    def reduce(self, other):
        self.res += other.res


def mapreduce(worker_class, input_class, config):
    workers = worker_class.create_workers(input_class, config)
    return execute(workers)


if __name__ == '__main__':
    sample_data()
    # print("res: ", map_reduce(data_dir=tmp_path))
    config = {'data_dir': tmp_path}
    print("res:", mapreduce(LineWorker, PathInputData, config))