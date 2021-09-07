# -*- coding:utf-8 -*-
import numpy as np
import time

#input data: a and b are numpy-array-like data
#output the data a and b cos similarity
#such as a:[[1 2]    b:[[4 5]
#           [3 4]]      [6 7]]
#return is[[ 0.97780241  0.9701425 ]
#          [ 0.99951208  0.99788011]]
#that is for a,it has 2 objects and 2 features ,for b the same
#return is for re[0,0] is a's 1st object and b's 1st object similarity
#re[0,1] is a's 1st object and b's 2sc object similarity
def cos(a,b):
    if(a.shape[1] != b.shape[1]):
        return 'the data a and data b must have same cols nums'
    else:
        matrix_a = np.matrix(a)
        matrix_b = np.matrix(b)
        mul = matrix_a*matrix_b.T
        norm_a = np.linalg.norm(a,axis=1)
        norm_b = np.linalg.norm(b,axis=1)
        norm = np.outer(norm_a,norm_b)
        cos = mul/norm
        return cos

a = np.random.random((10000,5))
b = np.random.random((100000,5))
start = time.time()
print(cos(a,b))
print('use %d seconds'%(time.time()-start))
