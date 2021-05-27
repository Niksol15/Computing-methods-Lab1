import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def polynomial_interpolation(x, y):
    assert (len(x) == len(y))
    x = np.asfarray(x)
    y = np.asfarray(y)

    arr_length = len(x)
    A = np.zeros((arr_length, arr_length))
    for i in range(arr_length):
        for j in range(arr_length):
            A[i][j] = x[i] ** j
    coefs = la.solve(A, y)

    def result(arg):
        res = 0
        for i in range(coefs.shape[0]):
            res += coefs[i] * (arg ** i)
        return res

    return result


def lagrange_interpolation(x, y):
    assert (len(x) == len(y))
    x = np.asfarray(x)
    y = np.asfarray(y)
    k = len(x)

    def l(arg, j):
        res = 1
        for i in range(k):
            if i != j:
                res *= (arg - x[i]) / (x[j] - x[i])
        return res

    def polynomial(arg):
        res = 0
        for i in range(k):
            res += y[i] * l(arg, i)
        return res

    return polynomial


def newton_interpolation(x, y):
    assert (len(x) == len(y))
    x = np.asfarray(x)
    y = np.asfarray(y)
    k = len(x)

    coef = np.zeros([k, k])
    coef[:, 0] = y
    for j in range(1, k):
        for i in range(k - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    a = coef[0]

    def polynomial(arg):
        res = 0
        n = np.ones(k)
        for i in range(1, k):
            n[i] = n[i - 1] * (arg - x[i - 1])
        for i in range(k):
            res += a[i] * n[i]
        return res

    return polynomial


def cubic_spline_interpolation(x, y):
    assert (len(x) == len(y))
    x = np.asfarray(x)
    y = np.asfarray(y)
    n = len(x)
    A = np.zeros((4 * n - 4, 4 * n - 4))
    b = np.zeros(4 * n - 4)
    '''
    A:  x[0]^3  x[0]^2 x[0] 1  0        0     0  0 ... 0
        x[1]^3  x[1]^2 x[1] 1  0        0     0  0 ... 0
        3x[1]^2 2x[1]  1    0 -3x[1]^2 -2x[1] -1 0 ... 0
        6x[1]   2      0    0 -6x[1]   -2     0  0 ... 0
        ...
        ...  x[i]^3      x[i]^2     x[i]     1 0            0          0  0 ... 0
        ...  x[i + 1]^3  x[i + 1]^2 x[i + 1] 1 0            0          0  0 ... 0
        ...  3x[i + 1]^2     2x[i + 1]      1        0 -3x[i + 1]^2 -2x[i + 1] -1 0 ... 0
        ...  6x[i + 1]       2x[i + 1]      0        0 -6x[i + 1]   -2         0  0 ... 0
        ...  
        ... x[n - 2]^3  x[n - 2]^2 x[n - 2] 1  0            0          0  0 ... 0
        ... x[n - 1]^3  x[n - 1]^2 x[n - 1] 1  0            0          0  0 ... 0
        ... 3x[n - 2]^2 2x[n - 2]  1        0 -3x[n - 1]^2 -2x[n - 1] -1  0 ... 0
        ... 6x[n - 1]   2          0        0  0            0          0  0 ... 0
        ... x[n - 1]^3 x[n - 1]^2 x[n - 1] 1
        6x[0] 2 0 ... 0
        ... 6x[n - 1]  2          0        0  0            0          0  0 ... 0
        
    b: y[0] y[1] 0 0 y[1] y[2] 0 0 ...  y[n - 1] 0 0 0
    x: a[0] b[0] c[0] d[0] ... a[n - 2] b[n - 2] c[n - 2] d[n - 2]    
    '''
    for i in range(n - 1):
        for j in range(4):
            A[4 * i][4 * i + j] = x[i] ** (3 - j)
            A[4 * i + 1][4 * i + j] = x[i + 1] ** (3 - j)
        b[4 * i] = y[i]
        b[4 * i + 1] = y[i + 1]
        if i != n - 2:
            A[4 * i + 2][4 * i] = 3 * (x[i + 1] ** 2)
            A[4 * i + 2][4 * i + 1] = 2 * x[i + 1]
            A[4 * i + 2][4 * i + 2] = 1
            A[4 * i + 2][4 * (i + 1)] = -3 * (x[i + 1] ** 2)
            A[4 * i + 2][4 * (i + 1) + 1] = -2 * x[i + 1]
            A[4 * i + 2][4 * (i + 1) + 2] = -1
            A[4 * i + 3][4 * i] = 6 * x[i + 1]
            A[4 * i + 3][4 * i + 1] = 2
            A[4 * i + 3][4 * (i + 1)] = -6 * x[i + 1]
            A[4 * i + 3][4 * (i + 1) + 1] = -2

        else:
            A[4 * i + 2][0] = 6 * x[0]
            A[4 * i + 2][1] = 2
            A[4 * i + 3][4 * (n - 2)] = 6 * x[n - 1]
            A[4 * i + 3][4 * (n - 2) + 1] = 2


    abcd = la.solve(A, b)

    def polynomial(arg):
        coef = np.searchsorted(x, arg, 'right') - 1
        if coef > n - 2 : coef = n - 2
        return abcd[4 * coef] * (arg ** 3) + abcd[4 * coef + 1] * (arg ** 2) + abcd[4 * coef + 2] * arg + abcd[ \
            4 * coef + 3]

    return polynomial
