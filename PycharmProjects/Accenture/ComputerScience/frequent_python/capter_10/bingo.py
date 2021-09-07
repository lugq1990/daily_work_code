# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import random
from Accenture.ComputerScience.frequent_python.capter_10.tombola import Tombola


class BingoCage(Tombola):
    def __init__(self, items):
        self._random = random.SystemRandom()
        self._items = items
        self.load(items)

    def load(self, items):
        self._items.extend(items)
        self._random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError("pick empty")

    def __call__(self, *args, **kwargs):
        self.pick()

