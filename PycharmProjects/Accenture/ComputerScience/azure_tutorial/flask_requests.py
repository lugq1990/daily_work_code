# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import array

@profile
def f_list():
    a = []
    a.extend(range(100000))
    return a


@profile
def f_array():
    a = array.array('l')
    a.extend(range(100000))
    return a


if __name__ == '__main__':
    f_list()
    f_array()