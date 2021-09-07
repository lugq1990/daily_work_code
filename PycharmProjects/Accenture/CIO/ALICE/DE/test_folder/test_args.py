# -*- coding:utf-8 -*-
"""This is write for args functions"""
# for * in the args for function means with we could put with *
def f(*args):
    # in fact you could also write a function in this function to get the parameter from the f() function
    def f_insight():
        # here is just to sum over the whole parameter in (*args)
        n = 0
        for i in args:
            # in fact, you could use check with the type in the insight function
            if not isinstance(i, int):
                raise TypeError('Here function just get with integer value!')
            n += i
        return n

    # here you could see that there are 2 functions, one is f(*args), the other is f_insight(),
    # f_insight() is the insight function to get the parameters of the outer function f()
    # here could just return a function!!! without the f_insight() with ()
    return f_insight

# here is the main function, whole step should run from here
if __name__ == '__main__':
    # here you have to know that the f() function return is a function!
    # so you have to make one variable to receive the return value
    # here you could also see that here you could add more parameters, not just one!!!!!!
    get_return_function = f(1, 2, 3, 1,)

    # you will see that here is just a function
    print(get_return_function)
    # here we want to get the real returned value, have to add () with returned function
    print('get result:', get_return_function())