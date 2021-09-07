# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""


class StackToQueue:
    def __init__(self):
        self.stack_a = []
        self.stack_b = []

    def push(self, data):
        self.stack_a.append(data)

    def pop(self):
        if not self.stack_b:
            if not self.stack_a:
                return None
            self.transfer()

        # self.show()

        return self.stack_b.pop()

    def transfer(self):
        while len(self.stack_a) != 0:
            d = self.stack_a.pop()
            self.stack_b.append(d)

    def show(self):
        print("stack b:")
        for i in range(len(self.stack_b)):
            print(self.stack_b[i], end=' ')

        print()

        print("stack a:")
        for i in range(len(self.stack_a)):
            print(self.stack_a[i], end=' ')
        print()


if __name__ == '__main__':
    stack = StackToQueue()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    stack.show()
    print("get ",stack.pop())
    stack.transfer()
    stack.show()
