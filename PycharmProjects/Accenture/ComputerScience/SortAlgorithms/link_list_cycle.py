# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""


class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        if self.head is None:
            self.head = Node(data)
        else:
            node = self.head
            while node.next is not None:
                node = node.next
            node.next = Node(data)

    def show(self):
        node = self.head
        while node is not None:
            print(node.value, end=' ')
            node = node.next


def is_cycle(node):
    p1 = node.head
    p2 = node.head
    while p2 is not None and p2.next is not None:
        p1 = p1.next
        p2 = p2.next.next
        if p1 == p2 :
            return True, p1, p2

        if p2 is None:
            p2 = node.head
        if p1 is None:
            p1 = node.head

    return False


def length_cycle(node):
    p1 = node.head
    p2 = node.head
    n = 0
    p1_length = 0
    p2_length = 0
    while p2 is not None and p2.next is not None:
        p1 = p1.next
        p2 = p2.next.next
        p1_length += 1
        p2_length += 2

        if p1 == p2:
            n += 1
        if p1 is None:
            p1 = node.head
        if p2 is None:
            p2 = node.head
        if n == 2:
            return p2_length - p1_length
    return None


if __name__ == '__main__':
    node = LinkList()
    node.insert(4)
    node.insert(3)
    node.insert(7)
    node.insert(4)
    node.show()

    print(is_cycle(node))
