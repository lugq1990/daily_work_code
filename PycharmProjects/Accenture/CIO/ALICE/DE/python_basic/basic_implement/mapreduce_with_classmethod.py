# -*- coding:utf-8 -*-
"""
This is more advance with class method implement

@author: Guangqiang.lu
"""
import tempfile
import os
from threading import Thread
import shutil

tmp_path = tempfile.mkdtemp()


def create_sample_data(n=3):
    for i in range(n):
        with open(os.path.join(tmp_path, "test_{}.txt".format(str(i))), 'w') as f:
            f.write("a" * (i + 1) * 100)


class GeneralInputData:
    def read(self):
        raise NotImplementedError

    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError


class LineInputData(GeneralInputData):
    def __init__(self, path):
        super(LineInputData, self).__init__()
        self.path = path

    def read(self):
        return open(self.path).read()

    @classmethod
    def generate_inputs(cls, config):
        data_dir = config['data_dir']
        for f in os.listdir(data_dir):
            yield cls(os.path.join(data_dir, f))


class GeneralWorker:
    def __init__(self, data):
        self.data = data
        self.res = None

    def map(self):
        pass

    def reduce(self, other):
        pass

    @classmethod
    def create_workers(cls, input_class, config):
        workers = []
        for input_data in input_class.generate_inputs(config):
            workers.append(cls(input_data))

        return workers


class LineWorkers(GeneralWorker):
    def map(self):
        data = self.data.read()
        self.res = data.count("a")

    def reduce(self, other):
        self.res += other.res


def execute(workers):
    threads = [Thread(target=w.map) for w in workers]
    for t in threads: t.start()
    for t in threads: t.join()

    first, others = workers[0], workers[1:]
    for w in others:
        first.reduce(w)
    return first.res


def mapreduce(worker_class, input_class, config):
    workers = worker_class.create_workers(input_class, config)
    return execute(workers)


if __name__ == '__main__':
    create_sample_data()
    print("Get: ", mapreduce(LineWorkers, LineInputData, {"data_dir": tmp_path}))
    shutil.rmtree(tmp_path)