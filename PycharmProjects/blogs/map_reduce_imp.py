import os 
from threading import Thread


class InputData:
    def read(self):
        raise NotImplementedError


class PathInput(InputData):
    def __init__(self, path):
        self.path = path

    def read(self):
        return open(self.path).read()


class Worker:
    def __init__(self, input_data):
        self.input_data = input_data
        self.result = None
    
    def map(self):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

class LineCounterWorker(Worker):
    def map(self):
        data = self.input_data.read()
        self.result = data.count('1')
    
    def reduce(self, other):
        self.result += other.result


def generate_inputs(data_dir):
    for name in os.listdir(data_dir):
        yield PathInput(os.path.join(data_dir, name))


def create_workers(input_list):
    workers = []
    for input_data in input_list:
        workers.append(LineCounterWorker(input_data))
    return workers


def execute(workers):
    threads = [Thread(target=w.map) for w in workers]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    first, rest = workers[0], workers[1:]
    for worker in rest: 
        first.reduce(worker)
    
    return first.result


def map_reduce(data_dir):
    inputs = generate_inputs(data_dir)
    workers = create_workers(inputs)
    return execute(workers)


import tempfile
import numpy as np

tmp_path = tempfile.mkdtemp()

for i in range(2):
    with open(os.path.join(tmp_path, 'test_{}.txt'.format(str(i))), 'w') as f:
        f.write(','.join([str(np.random.randint(10)) for _ in range(100)]))


if __name__ == '__main__':
    print(map_reduce(tmp_path))
