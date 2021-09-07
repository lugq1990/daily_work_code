# -*- coding:utf-8 -*-
"""the decorator is just not want to change the logic of one function, but to add more with this function or other"""

# here I want to add some features with f_basic() function
# you see for now, but here for now I also should write one function
def add_feature(func):
    # here is why the wrapper function with two parameters: *args, **kwargs
    # *args: we don't know how many parameters we will get, so with *args to ensure any length we will solve
    # *kwargs: in fact, we could use add the parameters with directory as key-value, so here to ensure!
    def wrapper(*args, **kwargs):
        print('Now is:')
        return func(*args, **kwargs)
    return wrapper

# here is basic function
def f_basic():
    # here I also import datetime module, just like numpy and pandas and ....
    import datetime
    print('{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# you see that in fact the two function is same in content! here add the decorator!
@add_feature
def f_advance():
    # here I also import datetime module, just like numpy and pandas and ....
    import datetime
    print('{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


# here is main function
if __name__ == '__main__':
    # use the basic function
    f_basic()
    print('*'*20)
    # here is decorator function, even with same logic, we could get more than basic
    f_advance()