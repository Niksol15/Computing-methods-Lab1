import edulinalg as ela
import numpy as np

if __name__ == '__main__':
    rnd_matrix = ela.hilbert(20)
    b = np.dot(rnd_matrix, np.ones(20))
    print(np.linalg.norm(np.dot(rnd_matrix, np.linalg.solve(rnd_matrix, b.transpose())) - b))
    print(np.linalg.norm(np.dot(rnd_matrix, ela.Seidel_method(rnd_matrix, b.transpose(), np.zeros_like(b))) - b))
    print(np.linalg.norm(np.dot(rnd_matrix, ela.lup_solve(rnd_matrix, b.transpose())) - b))