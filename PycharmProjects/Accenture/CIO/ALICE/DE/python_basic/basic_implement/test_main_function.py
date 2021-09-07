# -*- coding:utf-8 -*-
"""This is just to test how the python if __name__ == '__main__' function """


print("Now is f1")
def f1():
    print('hello world, Im f1')

print("now is f2")
def f2():
    print("hello world, Im' f2")

print("now is main function")
if __name__ == '__main__':
    f1()
    f2()