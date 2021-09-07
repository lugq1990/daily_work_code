# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
from pprint import pprint


class ToDictMinin:
    def to_dict(self):
        return self._traverse_dict(self.__dict__)

    def _traverse_dict(self, dict_instance):
        output = {}
        for k, v in dict_instance.items():
            output[k] = self._traverse(k, v)
        return output

    def _traverse(self, key, value):
        if isinstance(value, ToDictMinin):
            return value.to_dict()
        elif isinstance(value, dict):
            return self._traverse_dict(value)
        elif isinstance(value, list):
            return [self._traverse(key, i) for i in value]
        elif hasattr(value, '__dict__'):
            return self._traverse_dict(value.__dict__)
        else:
            return value


class BinaryTree(ToDictMinin):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class BinaryTreeParent(BinaryTree):
    def __init__(self, value, left=None, right=None, parent=None):
        super().__init__(value, left=left, right=right)
        self.parent = parent

    def _traverse(self, key, value):
        if isinstance(value, BinaryTreeParent) and key == 'parent':
            return value.value
        else:
            return super()._traverse(key, value)


class NameSubTree(ToDictMinin):
    def __init__(self, name, tree_with_parent):
        self.name = name
        self.tree_with_parent = tree_with_parent



if __name__ == '__main__':
    #tree = BinaryTree(10, left=BinaryTree(7, right=BinaryTree(8)), right=BinaryTree(12, left=BinaryTree(129)))
    tree = BinaryTreeParent(10)
    tree.left = BinaryTreeParent(1, parent=tree)
    tree.left.right = BinaryTreeParent(2, parent=tree.left)
    pprint(tree.to_dict())
    print("***")
    my_tree = NameSubTree('foo', tree.left)
    pprint(my_tree.to_dict())

