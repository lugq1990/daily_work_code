# -*- coding:utf-8 -*-
"""This is to implement the thread lock with threading module for two different function to
get the same source data as the global variable"""
import threading
import inspect
import time

# init the sub-class inherit from the threading.Thread class
class Thread(threading.Thread):
    def __init__(self, target, *args):
        super(Thread, self).__init__(target=target, args=args)
        self.start()

# global variable
count = 0
# lock with threading lock
lock = threading.Lock()

# incremental function
def incre():
    global count
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    print('Acquiring lock for function %s()'% caller)
    with lock:
        print('Functin %s() has get the lock!'%(caller))
        count += 1

def hello():
    while count < 5:
        incre()

def hello2():
    while count < 5:
        incre()

def hello3():
    while count < 5:
        incre()

def main():
    h1 = Thread(hello)
    h2 = Thread(hello2)
    h3 = Thread(hello3)

if __name__ == '__main__':
    main()
