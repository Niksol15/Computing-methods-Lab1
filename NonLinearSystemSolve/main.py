import numpy as np
import solver


def first_func(x):
    return np.array([x[0] ** 2 / x[1] ** 2 - np.cos(x[1]) - 2.0, x[0] ** 2 + x[1] ** 2 - 6.0])


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
                res[i] += x[j] ** 2 - j ** 2
            else:
                res[i] += (x[j] ** 3 - j ** 3)
    return res


def second_func_jacobian(x):
    n = len(x)
    jacobian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                jacobian[i][j] = 2 * x[j]
            else:
                jacobian[i][i] = 3 * (x[i] ** 2)
    return jacobian


if __name__ == '__main__':
    print('Newton\nFirst: ')
    print(first_func(solver.Newton_method(first_func, first_func_jacobian, np.ones(2))))
    print('Second: ')
    print(second_func(solver.Newton_method(second_func, second_func_jacobian, np.ones(7) * 5)))
    print('\n\nModified Newton\nFirst:')
    print(first_func(solver.Newton_modified_method(first_func, first_func_jacobian, np.ones(2))))
    print('Second:')
    print(second_func(solver.Newton_modified_method(second_func, second_func_jacobian, np.ones(7) * 5)))
    print('\n\nRelaxation\nFirst:')
    print(first_func(solver.relaxation(first_func, np.ones(2))))
    print('Second:')
    print(second_func(solver.relaxation(second_func, np.ones(7) * 5)))
