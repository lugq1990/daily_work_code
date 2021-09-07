# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import random
from Accenture.ComputerScience.frequent_python.capter_10.tombola import Tombola


@Tombola.register
class TomboList(list):
    def pick(self):
        if self:
            pos = random.randrange(len(self))
            return self.pop(pos)
        else:
            raise LookupError("empty")

    load = list.extend

    def loaded(self):
        return bool(self)

    def inspect(self):
        return tuple(sorted(self))

