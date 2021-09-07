# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import doctest
import os

from Accenture.ComputerScience.frequent_python.capter_10.tombola import Tombola
from Accenture.ComputerScience.frequent_python.capter_10 import bingo, lotto, fake, tombolist

cur_dir = os.path.abspath(os.curdir)

# test_file = os.path.join(cur_dir, "test.txt")
test_file = "tombola_test.rst"

test_msg = '{0:16} {1.attempted:2} tests, {1.failed:2} failed - {2}'

def main(argv):
    verbose = '-v' in argv
    real_subclass = Tombola.__subclasses__()
    vir_subclass = list(Tombola._abc_registry)

    for cls in real_subclass + vir_subclass:
        test(cls, verbose)

def test(cls, verbose=False):
    res = doctest.testfile(test_file, globs={"concreteTombalo": cls},
                           verbose=verbose, optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
    tag = "fail" if res.failed else 'ok'
    print(test_msg.format(cls.__name__, res, tag))


if __name__ == '__main__':
    import sys
    main(sys.argv)
