# -*- coding:utf-8 -*-
"""This is just to test the python unittest module for robust code """

import unittest


def convert(x):
    return x.upper()


class TestString(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('fOo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue("FOO".isupper())
        self.assertFalse('FOO'.islower())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

    def test_convert(self):
        x = 'aBC'
        self.assertEqual(convert(x), 'ABC')


if __name__ == '__main__':
    unittest.main()