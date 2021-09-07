# -*- coding:utf-8 -*-
"""


@author: Guangqiang.lu
"""
import collections

Grade = collections.namedtuple('Grade', ('score', 'weight'))


class Subject:
    def __init__(self):
        self.grades = []

    def report_grade(self, score, weight):
        self.grades.append(Grade(score, weight))

    def average_grade(self):
        total, total_weight = 0., 0.
        for grade in self.grades:
            total += grade.score * grade.weight
            total_weight += grade.weight
        return total / total_weight


class Student:
    def __init__(self):
        self.subjects = {}

    def subject(self, name):
        if name not in self.subjects:
            self.subjects[name] = Subject()
        return self.subjects[name]

    def average_grade(self):
        total, count = 0, 0
        for sub in self.subjects.values():
            total += sub.average_grade()
            count += 1
        return total / count


class GradeBook:
    def __init__(self):
        self.students = {}

    def student(self, name):
        if name not in self.students:
            self.students[name] = Student()
        return self.students[name]


if __name__ == '__main__':
    book = GradeBook()
    lu = book.student('lu')
    math = lu.subject('math')
    math.report_grade(80, .1)
    print(lu.average_grade())