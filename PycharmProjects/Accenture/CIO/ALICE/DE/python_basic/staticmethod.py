# -*- coding:utf-8 -*-
"""this is to implement both classmethod and staticmethod of a class"""

class Date(object):
    def __init__(self, year=0, month=0, day=0):
        self.year = year
        self.month = month
        self.day = day

    # here is the class method, if I write the class method, then if another class inherent from this class
    # then the child class with get this method! This method will return the Date object!
    # tips: classmethod must with first parameter wiht cl
    # if we want to construct an object from one string, here is means with overloading,
    # so for using classmethod, that we could instance the Date object with one string, with different
    # input string!
    @classmethod
    def date_from_string(cls, date_str):
        year, month, day = map(int, date_str.split('-'))
        date1 = cls(year, month, day)
        return date1

    @classmethod
    def date_from_string_with_sep(cls, date_str):
        if '-' in date_str:
            year, month, day = map(int, date_str.split('-'))
        elif ',' in date_str:
            year, month, day = map(int, date_str.split(','))
        else:
            year, month, day = date_str[:4], date_str[4:6], date_str[6:]

        # is called with class to instant the object
        date = cls(year, month, day)
        return date

    # here is one static method, this method could be called without instancing the object,
    # just with Date.is_valide(xx)
    @staticmethod
    def is_valide(date_str):
        year, month, day = map(int, date_str.split('-'))
        return year < 3000 and month < 13 and day < 32

if __name__ == '__main__':
    date = '2019-06-20'
    date_obj = Date.date_from_string(date)
    print('Get year:', date_obj.year)
    print(Date.is_valide(date))

    print('*'*10)
    date2 = '20190621'
    date2_obj = Date.date_from_string_with_sep(date2)
    print('get year:', date2_obj.year)