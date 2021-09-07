# -*- coding:utf-8 -*-
"""this is to implement the SVD decomposition, and reconstruct the original matrix with
decomposited matrix, U, V, W"""
import numpy as np

# here is just to create a random matrix
data = np.matrix(np.random.random((10, 7)))

# first to check the rank of the matrix
print('Matrix rank:{}'.format(np.linalg.matrix_rank(data)))

# compute the u, v, w to with SVD
u, v, w = np.linalg.svd(data)
print('Get decomposition:')
print('U matrix:', u)
print('V matrix:', v)
print('W matrix:', w)

# here is to restore the matrix with dot product
# first to make the diagonal matrix of V
v_dig = np.diag(v)
# recompute the V to satisfy the dot product, with the rank to add with the zero matrix with row
v_dig = np.concatenate((v_dig, np.zeros((u.shape[0] - np.linalg.matrix_rank(data), v_dig.shape[1]))), axis=0)

data_re = np.dot(np.dot(u, v_dig), w)

print('*'*30)
print('Here is to use SVD result to compute the User matrix and production matrix')
print('User matrix:', np.dot(u, v_dig))
print('Product matrix:', np.dot(v_dig, w))

print('Where two matrix closed to each other: ', np.allclose(data, data_re))


import zipfile
zipfile.ZipFile().extractall()