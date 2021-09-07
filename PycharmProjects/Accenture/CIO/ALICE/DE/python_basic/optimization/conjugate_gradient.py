# -*- coding:utf-8 -*-
"""This is to implement the conjugate gradient descent according to the wiki website for
explaining the algorithm, with this linK: https://en.wikipedia.org/wiki/Conjugate_gradient_method
as bellow code running result, that we could find that during just 10 steps, that we find that
the solution by using the conjugate solution is really close to the true solution!
Noted:
For AX=B, the constrains is that A is must symmetric and positive definite!"""

import numpy as np

# as conjugate gradient is an iterative algorithm, so here is one parameter to define when to stop
step = 10

# we are solving the AX=B linear function
a = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

# the solution for this x = [1/11, 7/11]
# before the loop, we have to pre-guess the x0
x_guess = [2, 1]
for i in range(step):
    # compute the residual with random guess x
    residual = b - np.dot(a, x_guess)
    # after compute the residual, compute the alpha
    alpha = np.dot(residual.T, residual) / (residual.T @ a @ residual)
    # update the guess x
    x_guess = x_guess + alpha * residual

x_true = [1/11, 7/11]

print("after %d step, final x is:", x_guess)
print("Truth x is: ", x_true)
print("how much difference between solution x and guess x:", (x_true - x_guess))