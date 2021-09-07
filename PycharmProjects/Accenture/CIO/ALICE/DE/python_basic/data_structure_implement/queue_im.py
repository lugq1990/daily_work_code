# -*- coding:utf-8 -*-
"""Just to implement the queue data structure, as I figure out that
we shouldn't use the list to implement the queue, as the remove step
will really slower for list as it will move the whole data with position changed.
The best way is use collections dequeue module"""


from collections import deque


class Queue:
    """Queue is first in first out."""

    def __init__(self):
        self.head = deque()

    def push(self, value):
        self.head.append(value)

    def pop(self):
        # revomve the first value from the list
        return self.head.popleft()

    def find(self, value):
        if value not in self.head:
            raise ValueError("Value %d not found in the list!" % value)
        return self.head.index(value)

    def insert(self, value, position=-1):
        if position < -1:
            position = -1
        self.head.insert(position, value)

    def remove(self, value):
        if value not in self.head:
            raise ValueError("Value %d not in the list!" % value)
        self.head.remove(value)

    def reverse(self):
        self.head = deque(list(reversed(self.head)))

    def delete(self):
        self.head = None

    @property
    def length(self):
        return len(self.head)

    def print_var(self):
        print('Get list:', self.head)


if __name__ == '__main__':
    q = Queue()
    q.push(0)
    q.push(1)
    q.push(2)
    q.insert(100, 1)
    q.print_var()

    q.remove(0)
    q.print_var()

    print('find position: ', q.find(1))