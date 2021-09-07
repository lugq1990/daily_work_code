# -*- coding:utf-8 -*-
print('hello world!')

def show(x):
    return 'Get value %s' % (str(x))

class Person(object):
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex

    def run(self):
        return "%s could run!" % (self.name)

    @property
    def sex(self):
        return self.sex