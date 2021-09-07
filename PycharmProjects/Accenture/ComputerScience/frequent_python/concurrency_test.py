# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import os
import shutil
from queue import Queue
from threading import Thread
from time import time
from multiprocessing.pool import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import warnings

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

warnings.simplefilter('ignore')


n_file = 10

try:
    os.mkdir('test_data')
except:
    pass


def write_file(n):
    # this is for IO bound test
    with open('test_data/test_{}.txt'.format(str(n)), 'w') as f:
        f.write(str(n) * 10000000)

    # this is for compute bound test
    # x, y =load_iris(return_X_y=True)
    # lr = LogisticRegression()
    # for i in range(10):
    #     lr.fit(x, y)


class Worker(Thread):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        while True:
            n = self.queue.get()
            try:
                write_file(n)
            except:
                pass
            finally:
                self.queue.task_done()   # we have to set the thread task to done!


def concur():
    queue = Queue()
    for i in range(8):
        worker = Worker(queue)
        worker.deamon = True
        worker.start()

    for i in range(n_file):
        queue.put(i)
    queue.join()    # thread lock


def common():
    for i in range(n_file):
        write_file(i)


def process():
    with Pool(8) as p:
        p.map(write_file, range(n_file))


def thread_poo():
    with ThreadPoolExecutor() as executor:
        executor.map(write_file, range(n_file))


def process_pool():
    with ProcessPoolExecutor() as executor:
        executor.map(write_file, range(n_file))

if __name__ == '__main__':
    start_time = time()
    common()
    end_time = time()
    print("with common {} seconds".format(end_time - start_time))

    print("*"* 10)

    start_time = time()
    concur()
    end_time = time()
    print("with concur {} seconds".format(end_time - start_time))

    print("*" * 10)

    start_time = time()
    process()
    end_time = time()
    print("with process {} seconds".format(end_time - start_time))

    print("*" * 10)

    start_time = time()
    thread_poo()
    end_time = time()
    print("with thread pool {} seconds".format(end_time - start_time))

    print("*" * 10)

    start_time = time()
    process_pool()
    end_time = time()
    print("with process pool {} seconds".format(end_time - start_time))
