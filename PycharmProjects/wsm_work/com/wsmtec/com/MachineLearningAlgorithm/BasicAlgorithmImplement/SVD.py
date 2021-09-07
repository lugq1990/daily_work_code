# -*- coding:utf-8 -*-
"""
    This is just used to compute the SVD using the eigenvalue and eigenvector of A.T*A and A*A.T
    svd(A) = U*V*W
    A:m*n, U:n*k, V:k*k, W:k*n
"""


import numpy as np

class svd(object):
    def __init__(self):
        pass

    def svd(self, data):
        # first compute the eigen value of a*a.T for V
        # eigen value and eigen vector of a*a.T
        eigen_value_n, eigen_vector_n = np.linalg.eig(np.dot(data, data.T))
        # eigen value and eigen vector for a.T*a
        eigen_value_tran, eigen_vector_tran = np.linalg.eig(np.dot(data.T, data))

        V = np.sqrt(eigen_value_n)
        U = eigen_vector_n
        W = eigen_vector_tran
        return U, V, W.T

# a = np.array([[3,4,5],[2,1,4]])
# u, v, w = svd().svd(a)
# u1, v1, w1 = np.linalg.svd(a)


"""
    This function is used to use the svd decomposite the data, and turn back original data using the u, v and w
"""
def svd_conver_original(data):
    # first use the svd to original data to u, v, w
    u, v, w = np.linalg.svd(data)
    # because the matrix maybe not square
    if data.shape[0] != data.shape[1]:
        print('This is a non-square matrix')
        v_new = np.concatenate((np.diag(v), np.zeros((u.shape[0], w.shape[0]-u.shape[1]))), axis=1)
    else:
        print('This is square matrix')
        v_new = v
    # now use dot product of u, v_new and w to build original data
    return np.dot(np.dot(u, v_new), w)

# a = np.random.random((2,4))
# print(a)
# print('*'*50)
# print(svd_conver_original(a))


"""
    This function is used to get the top k components using SVD
"""
import warnings

def get_top_k_components(data, k=2):
    if data.shape[0] < k or data.shape[1] < k:
        warnings.warn('not matched data shape, use smaller k please! ')
        return
    # compute the u, v and w
    u, v, w = np.linalg.svd(data)
    v[k : ] = 0
    v_new = np.concatenate((np.diag(v), np.zeros((u.shape[1], w.shape[0]-v.shape[0]))), axis=1)
    return np.dot(np.dot(u, v_new), w)

a = np.random.random((3,3))
print(a)
print('*'*50)
print(get_top_k_components(a, k=4))