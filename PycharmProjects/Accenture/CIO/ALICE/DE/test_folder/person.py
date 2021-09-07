# -*- coding:utf-8 -*-
"""This is a Person class implement to do set name and get name"""


def p():
    print('Hello!')


class Person(object):
    def __init__(self):
        self.name = []

    def set_name(self, name):
        self.name.append(name)
        return self

    def get_name(self, name_id):
        if name_id >= len(self.name):
            return 'No user'
        else:
            return self.name[name_id]


if __name__ == '__main__':
    # person = Person()
    # person.set_name('lu')
    # print('Get name: ', person.get_name(0))
    p()
    print(Person().set_name('lu').get_name(0))
