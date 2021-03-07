import numpy as np
from scipy.linalg import hilbert


def rnd_diagonally_dominant_matrix(low, high, size):
    matrix = np.random.uniform(low, high, (size, size))
    for i in range(size):
        if abs(matrix[i][i]) <= np.sum(np.absolute(matrix[i])) - abs(matrix[i][i]):
            matrix[i][i] = np.sign(matrix[i][i]) * (np.sum(np.absolute(matrix[i])) + np.random.uniform(1, 5))
    return matrix


def lup_decomposition(a):
    A = a
    n = A.shape[0]
    P = np.identity(n)
    for k in range(n):
        max_val = 0
        leading = k
        for i in range(k, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                leading = i
        A[k], A[leading] = A[leading], A[k].copy()
        P[k], P[leading] = P[leading], P[k].copy()
        for i in range(k + 1, n):
            A[i][k] /= A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= A[i][k] * A[k][j]
    L = np.identity(n)
    U = np.zeros(A.shape)
    for k in range(n):
        for j in range(k, n):
            U[k][j] = A[k][j]
        for i in range(k):
            L[k][i] = A[k][i]
    return P, L, U


def lup_solve(A, b):
    P, L, U = lup_decomposition(A)
    Pb = np.dot(P, b)
    n = A.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.sum(L[i][:i] * y[:i])
    x = np.zeros(n)
    for i in range(n - 1,  -1, -1):
        x[i] = (y[i] - np.sum(U[i][i + 1:] * x[i + 1:])) / U[i][i]
    return x


ITERATION_LIMIT = 10000


def Jacoby_method(A, b, init_x):
    x = init_x
    for it_counter in range(ITERATION_LIMIT):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.allclose(x_new, x):
            return x_new
        x = x_new
    return x


def Seidel_method(A, b, init_x):
    x = init_x
    for it_counter in range(ITERATION_LIMIT):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2)/ A[i, i]
        if np.allclose(x_new, x):
            print(it_counter)
            return x_new
        x = x_new
    return x

