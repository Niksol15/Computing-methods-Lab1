import numpy as np
import numpy.linalg


def first_func(x):
    return np.array([(x[0] ** 2 / x[1] ** 2 - np.cos(x[1]) - 2.0), (x[0] ** 2 + x[1] ** 2 - 6.0)])


def first_func_jacobian(x):
    res = np.zeros((2, 2))
    res[0][0] = 2 * x[0] / (x[1] ** 2)
    res[0][1] = np.sin(x[1]) - 2 * ((x[0] ** 2) / (x[1] ** 3))
    res[1][0] = 2 * x[0]
    res[1][1] = 2 * x[1]
    return res


def second_func(x):
    n = len(x)
    res = np.zeros_like(x)
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] = x[j] ** 4 - j ** 4
            else:
                res[i] = 1e-2 * (x[j] ** 3 - j ** 3)
    return res


def second_func_jacobian(x):
    n = len(x)
    jacobian = np.zeros((n, n))
    for i in range(n):
        jacobian[i][i] = 1e-2 * 3 * (x[i] ** 2)
    return jacobian


def Newton_method(func, jacobian, approach, eps=1e-15):
    inv_jacobian = numpy.linalg.inv(jacobian(approach))
    next_it = approach - np.dot(inv_jacobian, func(approach))
    #while not np.allclose(next, approach):
    while abs(np.linalg.norm(next_it - approach)) > eps:
        approach = next_it.copy()
        inv_jacobian = numpy.linalg.inv(jacobian(approach))
        next_it = approach - np.dot(inv_jacobian, func(approach))

    return next


def Newton_modified_method(func, jacobian, approach, eps=1e-15):
    inv_jacobian = numpy.linalg.inv(jacobian(approach))
    next_it = approach - np.dot(inv_jacobian, func(approach))
    #while not np.allclose(next, approach):
    while abs(np.linalg.norm(next_it - approach)) > eps:
        approach = next_it.copy()
        next_it = approach - np.dot(inv_jacobian, func(approach))

    return next_it


def relaxation(func, approach, tau=5e-5, eps=1e-5):
    next_it = approach - tau * func(approach)
    while abs(np.linalg.norm(next_it - approach)) > eps:
        approach = next_it.copy()
        next_it = approach - tau * func(approach)
    return next_it
