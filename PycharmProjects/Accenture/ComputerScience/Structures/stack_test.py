# -*- coding:utf-8 -*-
"""This is to implement the stack: last in first out"""

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        return self.stack.pop()

    def print_stack(self):
        print('Get stack: ', self.stack)

    @property
    def length(self):
        return len(self.stack)


if __name__ == '__main__':
    stack = Stack()
    stack.push(0)
    stack.push(1)
    stack.push(2)

    stack.print_stack()
    stack.pop()
    print('Changed:')
    stack.print_stack()