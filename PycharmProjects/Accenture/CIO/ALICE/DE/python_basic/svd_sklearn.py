# -*- coding:utf-8 -*-
"""This function is based on sklearn implement with SVD, here could compare with sklearn svd"""
from sklearn.decomposition import TruncatedSVD
import numpy as np

data = np.random.random((4, 3))

# here is based on sklearn svd
model = TruncatedSVD(n_components=2)
model.fit(data)

v_from_sk = model.components_

# here just use numpy linear algebra to compute the svd
u, sigma, v = np.linalg.svd(data)
v_new = v[:2, :]

# here is just to get the max abs value's sign to multiply with the v_new
signs = np.sign(v_new[range(v_new.shape[0]), np.argmax(abs(v_new), axis=1)])
v_new *= signs[:, np.newaxis]

print('v from model:', v_from_sk)
print('v from dot: ', v_new)
print('here is to compute the V matrix',np.allclose(v_new, v_from_sk))

print('*' * 20)

u_from_sk = model.transform(data)
u_new = u[:, :2]
signs_u = np.sign(u_new[np.argmax(u_new, axis=0), range(u_new.shape[1])])
u_new *= signs_u
print('model result: ',u_from_sk)
print('dot product: ', u_new * sigma[:2])
print('here is to compute the U matrix', np.allclose(u_from_sk, u_new * sigma[:2]))

