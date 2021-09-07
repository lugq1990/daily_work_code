import numpy as np

a = np.array([[2, 4],
       [1, 3],
       [0, 0],
       [0, 0]])


u, v, w = np.linalg.svd(a)

# by defaul, u@u.T is unit matrix, same with w
print(u @u.T)
print(w @ w.T)

# for u is the eigen vector of a @a.T
_, u_new = np.linalg.eig(a @ a.T)
print(np.allclose(u, u_new))

_, w_new = np.linalg.eig(a.T @ a)
print(np.allclose(w, w_new))

# for v is the root square of eigen values
v_s, _ = np.linalg.eig(a @ a.T)
print(np.allclose(np.sqrt(v_s)[:len(v)], v))