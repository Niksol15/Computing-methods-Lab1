import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

A = 0
B = 15
N = 9


def func(_x):
    return np.sin(_x)


def my_interpolation(_x, _y):
    if len(_x) != len(_y):
        print("x.shape != y.shape")
        return 0
    arr_length = len(_x)
    _A = np.zeros((arr_length, arr_length))
    for i in range(arr_length):
        for j in range(arr_length):
            _A[i][j] = _x[i] ** j
    _coefs = la.solve(_A, y)

    def result(_arg):
        res = 0
        for i in range(_coefs.shape[0]):
            res += _coefs[i] * (_arg ** i)
        return res
    return result


x = np.linspace(A, B, N)
y = [func(i) for i in x]
finded_func = my_interpolation(x, y)

new_x = np.linspace(A, B, 10 * N)
finded_y = [finded_func(i) for i in new_x]
real_y = [func(i) for i in new_x]
plt.scatter(x, y)
plt.plot(new_x, real_y, "k")
plt.plot(new_x, finded_y, "r")
plt.show()

