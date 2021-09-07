# -*- coding:utf-8 -*-
"""This is used to test the class inherit from parent class with child class"""

# parent class
class Person(object):
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex

    def run(self, how_long=10):
        return '%s could run %d minutes'%(self.name, how_long)

    def eat(self, food='banana', frequence=4):
        return '%s like eating %s about %d days'%(self.name, food, frequence)

    @property
    def get_name(self):
        return self.name

# subclass
class Student(Person):
    def __init__(self, name, sex, num_class):
        super(Student, self).__init__(name=name, sex=sex)
        self.num_class = num_class

    def study(self, subject):
        return '%s good at %s'%(self.name, subject)


if __name__ == '__main__':
    ming = Student('ming', 'female', 3)
    print(ming.study('math'))
    print(ming.run())
    print('Get name :',ming.get_name)