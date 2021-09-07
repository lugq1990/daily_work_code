# -*- coding:utf-8 -*-
"""This is to implement the Queue: first in last out"""

class Queue:
    def __init__(self):
        self.queue = []

    def push(self, value):
        self.queue.append(value)

    def pop(self):
        return self.queue.remove(self.queue[0])

    def print_queue(self):
        print('Get queue:', self.queue)

    @property
    def length(self):
        return len(self.queue)


if __name__ == '__main__':
    queue = Queue()
    queue.push(0)
    queue.push(1)
    queue.push(2)

    queue.print_queue()
    queue.pop()
    print('Changed:')
    queue.print_queue()