# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from multiprocessing import Pool, Process, Lock
import os
import multiprocessing as mp

def f(x):
    print(x**2)


def info(title):
    print(title)
    print("Module name:", __name__)
    print("target process", os.getppid())
    print(os.getpid())

def f(q):
    q.put([1, 23, 'df'])

def l(l, i):
    l.acquire()
    try:
        print('get ', i)
    finally:
        l.release()

def not_l(i):
    print("GET", i)

if __name__ == '__main__':
    # with Pool(4) as p:
    #     print(p.map(f, [1, 2, 3, 4]))

    lock = Lock()
    # for i in range(10):
    #     p = Process(target=l, args=(lock, i))
    #
    #     p.start()

    for i in range(10):
        Process(target=not_l, args=(i ,)).start()
