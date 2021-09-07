# -*- coding:utf-8 -*-
"""This is to test the python multiprocessing module!"""

import multiprocessing
import os
import random
import time


"""This is basic multiprocessing"""
def hello(n):
    time.sleep(random.randint(1, 3))
    print("hello: {}".format(n))


"""Important thing for multi-processing means that for each thread will share with each global variable, 
so for each thread to change the global variable, then we should lock the variable!"""

my_list = []


def hello_global(n):
    global my_list
    time.sleep(random.randint(1, 3))
    my_list.append(os.getpid())
    print("Hello: {}".format(n))
    print("For thread %s with data: %s" % (os.getpid(), str(my_list)))

# have to set out the main!
# q = multiprocessing.Queue()
#
# def hello_in_main(n):
#     print("[{}]: hello".format(n))
#     q.put(os.getpid())


def s(x):
    info()
    return x**2


def info():
    print("Current thread:", os.getpid())


def foo(q, n):
    q.put("[{}]:hello".format(n))


def f(conn, data):
    conn.send(data)
    conn.close()


if __name__ == '__main__':
    # # the most basic use, without order
    # for i in range(5):
    #     p = multiprocessing.Process(target=hello, args=(i, ))
    #     p.start()


    # to ensure the whole process finished, then run others, just with join
    # processes = []
    # for i in range(5):
    #     p = multiprocessing.Process(target=hello, args=(i, ))
    #     processes.append(p)
    #     p.start()
    #
    # for one_p in processes:
    #     one_p.join()
    #
    # print("Done!")


    # test for the thread change global variable
    # as there isn't variable to be appended. as different thread use different variable
    # for each global thread! and the global variable unchanged, as python data structure is
    # not thread-safe!
    # processes = []
    # for i in range(5):
    #     p = multiprocessing.Process(target=hello_global, args=(i, ))
    #     processes.append(p)
    #     p.start()
    #
    # for one_p in processes:
    #     # join thread one by one
    #     one_p.join()
    #
    # print("Done!")
    # print("global variable:", my_list)


    """as python data structure is not safe, so the only safe that we could use is Queue(first in first out)
    queue could bridge the connection between thread and main threads"""
    # processes = []
    # for i in range(5):
    #     p = multiprocessing.Process(target=hello_in_main, args=(i, ))
    #     processes.append(p)
    #     p.start()
    #
    # for one_q in processes:
    #     one_q.join()
    #
    # res_list = []
    #
    # while not q.empty():
    #     res_list.append(q.get())
    #
    # print('Done!')
    # print("GET data:", res_list)


    """Pool of multiprocessing, we have to set the function out of main function!"""
    # with multiprocessing.Pool(5) as p:
    #     print(p.map(s, [2, 3, 5]))


    """set starter of thread!"""
    # multiprocessing.set_start_method('spawn')
    # q = multiprocessing.Queue()
    # for i in range(3):
    #     p = multiprocessing.Process(target=foo, args=(q, i))
    #     p.start()
    #     p.join()
    #
    # while not q.empty():
    #     print("Get data:", q.get())


    """Test thread connection with pipe, in fact, queue could do this!"""
    parent_conn, child_conn = multiprocessing.Pipe()
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=f, args=(child_conn, i))
        processes.append(p)
        p.start()
        print("Get data:", parent_conn.recv())

    for one_p in processes:
        one_p.join()











