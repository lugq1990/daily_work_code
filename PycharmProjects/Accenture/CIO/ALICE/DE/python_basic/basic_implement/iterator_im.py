# -*- coding:utf-8 -*-
"""This is just to implement the iterator object.
if we want to implement the iterator object, what we need is
to overwrite the object function: __iter__ to init and __next__ to get the value.
In fact, if we want to make a list to an iterator, then just use iter(list_object),
then we could just to get the data with next... just like deep learning training step
with batch_size to get the training data!"""


class Power:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.value:
            self.n += 1
            return 2 ** self.n
        else:
            raise StopIteration


if __name__ == '__main__':
    p = Power(10)
    for i in p:
        print('Get %d ' % i)