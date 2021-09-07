# -*- coding:utf-8 -*-
"""This is to compare the python unittest and py.test with same function"""

# this is unittest
# import unittest
#
# class TestEqual(unittest.TestCase):
#
#     def test_equal(self):
#         self.assertEqual(1, 1)
#
# if __name__ == '__main__':
#     unittest.main()


# This is py.test
def test_equal():
    assert 1 == 1

