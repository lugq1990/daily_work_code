# -*- coding:utf-8 -*-
import numpy as np

def solver(np_a,np_b):
    A = np.matrix(np_a)
    B = np.matrix(np_b)
    if(np_a.shape[1] == np_b.shape[0]):
        re = np.linalg.inv(A) * B.T
        return re
    else:
        return "the numpy array a and array b must have the same col-index nums!"

a = np.array([[401,-201],[-800,401]])
b = np.array([200,-200])
X = solver(a,b)
print("the linear equation solver result is ")
print(X)

