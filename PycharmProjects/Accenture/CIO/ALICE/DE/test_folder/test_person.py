# -*- coding:utf-8 -*-
"""This is used to test the person class by using unittest"""

import unittest
from .person import Person

name_list = []
id_list = []

class TestPerson(unittest.TestCase):
    """
    One note: if we want to make the unittest to test the function be line, then we should just make the test
    function with series number.
    """

    person = Person()

    def test_0_set(self):
        print("Start to test the set function")
        for i in range(4):
            name_list.append('name' + str(i))
            user_id = self.person.set_name('name' + str(i))
            self.assertIsNotNone(user_id)
            id_list.append(user_id)

    def test_1_get(self):
        print("Start to test the get function")

        length = len(name_list)

        for i in range(7):
            if i < length:
                self.assertEqual(name_list[i], self.person.get_name(i))
            else:
                self.assertEqual('No user', self.person.get_name(i))

if __name__ == '__main__':
    unittest.main()


from sklearn.tree import DecisionTreeClassifier

