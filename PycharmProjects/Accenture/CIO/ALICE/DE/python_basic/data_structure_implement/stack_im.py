# -*- coding:utf-8 -*-
"""This is to implement the stack data structure using the list!"""


class Stack:
    """Stack is Last in First out!"""
    def __init__(self):
        self.head = []

    def push(self, value):
        self.head.append(value)

    def pop(self):
        if len(self.head) == 0:
            raise ValueError('Nothing to be removed!')

        return self.head.pop()

    def delete(self, value):
        if value not in self.head:
            raise ValueError("Value %d not in the list!" % value)
        self.head.remove(value)

    def detele(self):
        self.head = None

    def find(self, value):
        if value not in self.head:
            raise ValueError("Value %d not in the list!" % value)

        return self.head.index(value)

    def insert(self, value, position=-1):
        if position < -1:
            position = -1

        self.head.insert(position, value)

    def reverse(self):
        self.head = list(reversed(self.head))

    @property
    def length(self):
        return len(self.head)

    def print_var(self):
        print('Get List: ', self.head)


if __name__ == '__main__':
    stack = Stack()
    stack.push(0)
    stack.push(1)
    stack.insert(2, -1)
    stack.insert(3, 1)
    stack.print_var()

    stack.delete(0)
    stack.print_var()

    stack.reverse()
    stack.print_var()

    print('Find: ', stack.find(1))