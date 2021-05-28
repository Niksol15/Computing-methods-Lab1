import numpy as np
import numpy.linalg


def Newton_method(func, jacobian, approach, eps=1e-15):
    inv_jacobian = numpy.linalg.inv(jacobian(approach))
    next_it = approach - np.dot(inv_jacobian, func(approach))
    while abs(np.linalg.norm(next_it - approach)) > eps:
        print(next_it)
        approach = next_it.copy()
        inv_jacobian = numpy.linalg.inv(jacobian(approach))
        next_it = approach - np.dot(inv_jacobian, func(approach))

    return next_it


def Newton_modified_method(func, jacobian, approach, eps=1e-15):
    inv_jacobian = numpy.linalg.inv(jacobian(approach))
    next_it = approach - np.dot(inv_jacobian, func(approach))
    while abs(np.linalg.norm(next_it - approach)) > eps:
        approach = next_it.copy()
        next_it = approach - np.dot(inv_jacobian, func(approach))

    return next_it


def relaxation(func, approach, tau=5e-5, eps=1e-15):
    next_it = approach - tau * func(approach)
    while abs(np.linalg.norm(next_it - approach)) > eps:
        approach = next_it.copy()
        next_it = approach - tau * func(approach)
    return next_it
