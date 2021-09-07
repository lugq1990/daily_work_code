# -*- coding:utf-8 -*-
"""This is to implement the double linked list"""

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

class DoubledLinkedList:
    def __init__(self):
        self.head = None

    def add(self, value):
        if self.head is None:
            self.head = Node(value)
        else:
            # Here just add value with the right side
            node = self.head
            while node.right is not None:
                node = node.right
            new_node = Node(value)
            node.right = new_node
            new_node.left = node

    def print_list(self):
        if self.head is None:
            print("Nothing in the linked list")
        else:
            node = self.head
            out = [node.value]
            while node.right is not None:
                node = node.right
                out.append(node.value)
            print('Get list:', out)

    @property
    def length(self):
        if self.head is None:
            return 0
        else:
            node = self.head
            n = 1
            while node.right is not None:
                node = node.right
                n += 1
            return n

if __name__ == '__main__':
    dll = DoubledLinkedList()
    dll.add(0)
    dll.add(1)
    dll.add(2)

    dll.print_list()
    print("Length:", dll.length)