# -*- coding:utf-8 -*-
"""here is just to show that for a function, that we could pass one or more parameters, or
even not pass parameters.
One more word about python function: why we use function to do something? in fact that if the logic is really
comprehensive or bigger then you think, so you could just write your logic in the function to do what you want!
maybe just sum over some list or get something from website etc.
"""
# import some module here
import numpy as np

# function without any parameter, I want to sum over 1 to 10 and return the result
def func_without_param():
    a = np.arange(1, 11)
    # here you could use foor loop or just use sum function
    # here is for loop
    n = 0
    for i in a:
        n += i

    # here with sum function
    n_sum = sum(a)
    # here should return the result of the function, you could return anything you want!
    # I return a tuple, right? you will see in main function
    return n, n_sum

# here I get make the function to get one parameter
# you have to know that: you have to give the parameter data with value when call the function
def with_one_param(data):
    return sum(data)

# you could put more than one parameter, but you could give the parameter value before you call it,
# means that default value of type is 'list'
def with_two_param(data, type='list'):
    if type == 'list':
        return sum(data)
    else:
        return sum(data.tolist())

# you could give the function with directory
def with_dire(dic):
    # to get the key and value with for loop
    for k, v in dic.items():
        print('Get key: {}, get value: {}'.format(str(k), str(v)))

if __name__ == '__main__':
    without_params_return = func_without_param()
    print('Type of the return: ', type(without_params_return).__name__)
    # But you want to get the return value separately
    without_re_1, without_re_2 = without_params_return[0], without_params_return[1]
    print('for loop result: {}, sum result: {}'.format(without_re_1, without_re_2))

    print('*'*20)
    print('with one parameter')
    a = np.arange(1, 11)
    print(with_one_param(a))

    print('*'*20)
    print('with two parameter')
    print(with_two_param(a))
    print(with_two_param(a, type='array'))

    print('*'*20)
    print('here is directory as parameter')
    direc = dict({'lu': 20, 'liu': 21})
    with_dire(direc)

    import functools
    print('*'*20)
    fix_f = functools.partial(with_two_param)
    print(fix_f(a))
    print(fix_f(a*2))
