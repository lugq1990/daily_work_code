# -*- coding:utf-8 -*-
"""This class is to implement the binary tree"""

class Node(object):
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

class BineryTree(object):
    def __init__(self):
        self.root = None

    def add(self, v):
        if self.root is None:
            self.root = Node(v)
        else:
            self._add(v, self.root)

    def _add(self, v, node):
        if v < node.value:
            if node.left is not None:
                self._add(v, node.left)
            else:
                node.left = Node(v)
        else:
            if node.right is not None:
                self._add(v, node.right)
            else:
                node.right = Node(v)

    def find(self, v):
        if self.root is not None:
            res = self._find(v, self.root)
            return res
        else:
            return 'no value'

    def _find(self, v, node):
        if v == node.value:
            f = node.value
            return f
        elif (node.left is not None and v < node.value):
            return self._find(v, node.left)
        elif (node.right is not None and v > node.value):
            print('node:', node.value)
            return self._find(v, node.right)

if __name__ == '__main__':
    tree = BineryTree()
    tree.add(0)
    tree.add(4)
    tree.add(1)
    # print('Get', tree.find(0))
    print(tree.root.right.left.value)
    print('Now is: ', tree.find(4))
