# -*- coding:utf-8 -*-

def f1(a):
    if a > 10:
        return a
    else:
        return 1

def f2(m):
    if m > 10:
        return 'Get value from f1() with upper to 10'
    else:
        return 'Get value lower than 10'

if __name__ == '__main__':
    f1_return = f1(11)
    f2(f1_return)   # just with return we couldn't see anything

    print('*'*20)
    f2_re = f2(f1_return)
    print(f2_re)   # here is f2 return result based on f1() function

    from sklearn.decomposition import PCA, TruncatedSVD
