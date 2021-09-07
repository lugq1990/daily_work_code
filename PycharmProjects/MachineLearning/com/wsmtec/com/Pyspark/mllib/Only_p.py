# -*- coding:utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression

num  = 10000000
x = np.random.random((num, 10))
y = np.random.randint(10, size=(num, 1))

lr = LogisticRegression()
model = lr.fit(x, y)
print('Model accuracy is %.5f'%(model.score(x, y)))

def f(x):
    return 1/(1+np.exp(x))
a = np.arange(100000)
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)
# this use the multi thread pool for the function
res = pool.map(f, a)

def co_pool():
    s_t = time.time()
    pool = ThreadPool(4)
    pool.map(f, a)
    e_t = time.time()
    print('Total time %.10f'%(e_t - s_t))

import time
def co():
    s_t = time.time()
    f(a)
    e_t = time.time()
    print('Total time %.10f'%(e_t - s_t))

def co_m(max_workers=100):
    s_t = time.time()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    pool.map(f, a)
    e_t = time.time()
    print('Total time %.10f' % (e_t - s_t))

import threading
workers = []
for _ in range(5):
    t = threading.Thread(target=f, args=(a, ))
    workers.append(t)
# start the thread
for thread in workers:
    thread.start()
    thread.join()

# build a thread pool using the future library
from concurrent.futures import ThreadPoolExecutor
excutor = ThreadPoolExecutor(max_workers=10)
excutor.submit(f, a)
def co_p():
    s_t = time.time()
    ex = ThreadPoolExecutor(max_workers=7)
    ex.submit(f, a)
    e_t = time.time()
    print('Total time %.10f' % (e_t - s_t))


np.linalg.eig(a)