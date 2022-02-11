"""Python 多线程

同时启用两个任务，利用多线程分别对全部的方法进行启动，如果只是正常的串行则需要5s，但通过多线程可以变成3s。

这个比较简单地讲明: https://www.cnblogs.com/fnng/p/3670789.html
"""
from time import ctime
import time
from threading import Thread


def read():
    print("Reading at ", ctime())
    time.sleep(2)
    print("end read")


def music():
    print("Listening at ", ctime())
    time.sleep(3)
    print('end music')
    

def evaluate_time(*func):
    start_time = time.time()
    for f in func:
        f()
    end_time = time.time()
    print("Use {} seconds".format(end_time - start_time))


if __name__ == '__main__':
    start_time = time.time()
    threads = []
    t1 = Thread(target=read)
    t2 = Thread(target=music)
    
    threads.append(t1)
    threads.append(t2)
    
    for t in threads:    
        # 启动每个线程，线程是同时启动的
        t.start()
    
    for t in threads:
        # 阻塞每一个线程，直到全部都完成
        t.join()
    # evaluate_time(read, music)
    end_time = time.time()
    print("Use {} seconds".format(end_time - start_time))
    