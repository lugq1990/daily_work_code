# -*- coding:utf-8 -*-

"""
    This is the BinarySearch Tree algorithm implement in Python
"""
import numpy as np

class BinarySearch:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

    def insert_node(self, value):
        if value <= self.value and self.left_child:
            self.left_child.insert_node(value)
        elif value <= self.value:
            self.left_child = BinarySearch(value)
        elif value > self.value and self.right_child:
            self.right_child.insert_node(value)
        else:
            self.right_child = BinarySearch(value)

    # Binary search whether the value exits
    def find_node(self, value):
        if value < self.value and self.left_child:
            return self.left_child.find_node(value)
        elif value > self.value and self.right_child:
            return self.right_child.find_node(value)

        return value == self.value

a = BinarySearch(10)
a.insert_node(1)
a.insert_node(20)

v = 2
print('Find %d '%(v)+np.str(a.find_node(v)))

