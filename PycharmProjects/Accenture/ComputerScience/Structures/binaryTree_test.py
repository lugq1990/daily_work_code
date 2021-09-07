# -*- coding:utf-8 -*-
"""This is to implement the binary tree algorithm"""

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.head = None
        self.numbers = 0

    def add(self, value, node=None):
        self.numbers += 1
        if self.head is None:
            self.head = Node(value)
        else:
            if node is None:
                node = self.head
            self._add(value, node)

    def _add(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                node = node.left
                return self._add(value, node)
        elif value > node.value:
            if node.right is None:
                node.right = Node(value)
            else:
                node = node.right
                return self._add(value, node)
        else:
            return "Couldn't add with same value!"


    def find(self, value, node=None):
        if self.head is None:
            raise ValueError("The tree is't constructed!")
        else:
            if node is None:
                node = self.head

            if value < node.value:
                if node.left is None:
                    return "Don't find the value, as value is lower than node value, but left node is None!"
                else:
                    node = node.left
                    return self.find(value, node)
            elif value > node.value:
                if node.right is None:
                    return "Don't find the value, as value is upper than node value, but right node is None!"
                else:
                    node = node.right
                    return self.find(value, node)
            else:
                return value

    def print_tree(self):
        if self.head is not None:
            self._print_value(self.head)

    def _print_value(self, node):
        if node is not None:
            if node == self.head:
                print('head:')
                print(node.value)
            else:
                print(node.value)

            if node.left is not None:
                print('left:')
                self._print_value(node.left)
            if node.right is not None:
                print('right:')
                self._print_value(node.right)

    @property
    def node_numbers(self):
        return self.numbers

    def delete(self, value):
        if self.head is None:
            return "The tree dosen't created!"
        else:
            self._delete(value)

    # in fact, I haven't tested this function as I will go to japan~
    def _delete(self, value, node=None):
        if node is None:
            node = self.head

        if value == node.value:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                # in fact, for one node, the left value is must smaller than the right node
                node = node.left

            self._delete(value, node)



if __name__ == '__main__':
    binary_tree = BinaryTree()
    binary_tree.add(0)
    binary_tree.add(-1)
    binary_tree.add(1)
    binary_tree.add(2)

    binary_tree.print_tree()
    print("find:", binary_tree.find(1))
    print("there are %d nodes"%(binary_tree.node_numbers))



from sklearn.linear_model import RidgeClassifier