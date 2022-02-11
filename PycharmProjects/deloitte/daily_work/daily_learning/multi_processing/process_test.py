"""python 多进程

多进程是启动两个进程在同时启动，两个进程互相不打扰,如果多进程会生成多个id，单进程则只有一个id。

比较清楚的说明多线程：https://mathpretty.com/9400.html
"""
from multiprocessing import Pool
import time
import os


def say(user):
    time.sleep(1)
    print(time.ctime())
    if isinstance(user, list):
        user = user[0]
    return user + ' Says. Thread ID :' + str(os.getpid())


if __name__ == '__main__':
    # 对比多进程和普通对比
    start_time = time.time()
    # 多进程, 时间1.2s
    p = Pool(2)
    
    # 必须两个list，每个对象变成arg
    users = [['lu'], ['liu']]
    res = p.starmap(say, users)
    print(res)
    
    end_time = time.time()
    print("Use {} seconds".format(end_time - start_time))
    
    # 单进程
    start_time = time.time()
    # map时间比循环时间短1.3s
    # res_sin = map(say, users)
    # 循环需要2s
    for u in users:
        print(say(u))
    
    end_time = time.time()
    print("Use {} seconds".format(end_time - start_time))
    
    