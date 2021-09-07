# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import random
from Accenture.ComputerScience.frequent_python.capter_10.tombola import Tombola


class Lottery(Tombola):
    def __init__(self, iterable):
        self._balls = list(iterable)

    def laod(self, iterable):
        self._balls.extend(iterable)

    def pick(self):
        try:
            pos = random.randrange(len(self._balls))
        except ValueError:
            raise LookupError("empty")
        return self._balls.pop(pos)

    def loaded(self):
        return bool(self._balls)

    def inspect(self):
        return tuple(sorted(self._balls))