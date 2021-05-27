import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import interpolate
import my_interpolate as interplt

A = 0
B = 5
N = 10
SCALE = 100


def func(x):
    return np.sin(3 * x) + np.cos(x)
    #return x

if __name__ == '__main__':
    x = np.linspace(A, B, N)
    y = func(x)
    plt.scatter(x, y)
    #polynomial_func = interplt.polynomial_interpolation(x, y)
    #lagrange_func = interplt.lagrange_interpolation(x, y)
    #newton_func = interplt.newton_interpolation(x, y)
    spline_func = interplt.cubic_spline_interpolation(x, y)
    new_x = np.linspace(A, B, SCALE * N)
    #polynomial_y = polynomial_func(new_x)
    #lagrange_y = [lagrange_func(i) for i in new_x]
    #newton_y = [newton_func(i) for i in new_x]
    spline_y = [spline_func(i) for i in new_x]
    real_y = func(new_x)

    plt.plot(new_x, real_y, "k")
    plt.plot(new_x, spline_y, 'r')
    #plt.plot(new_x, lagrange_y, "r")
    #plt.plot(new_x, newton_y, "b")

    plt.show()


