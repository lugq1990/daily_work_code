# -*- encoding: utf-8 -*-
'''
Training test with my new keyboard.

@time: 2021/05/31 14:36:20
@author: Guangqiang.lu
'''
def test_static_new(c=1):
    print(c)

class A:
    def __init__(self, val):
        self.val = val

    def test(self ):
        print(self.val)

    @staticmethod
    def test_static(b=1):
        print(b)



# This is used for the real test for what we have could do with my new keyboard.
# let's do the test as we want.
import os
from pathlib import Path

path = Path(__file__).parent

os.chdir(path)

print("current path:", path)
print(os.listdir(path))
print(os.path.abspath(os.curdir))


from sklearn.ensemble import GradientBoostingClassifier
