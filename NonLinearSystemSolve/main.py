import numpy as np
import solver

if __name__ == '__main__':
    #print(solver.first_func(solver.Newton_modified_method(solver.first_func, solver.first_func_jacobian, np.ones(2))))
    #print(solver.second_func(solver.Newton_modified_method(solver.second_func, solver.second_func_jacobian, np.ones(10))))
    #print(solver.second_func(solver.Newton_modified_method(solver.second_func, solver.second_func_jacobian, np.ones(30) * 100)))
    print(solver.second_func(solver.relaxation(solver.second_func, np.ones(100) * 0.1)))
