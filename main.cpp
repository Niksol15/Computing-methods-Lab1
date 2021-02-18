#include <iostream>
#include "CompMethods.hpp"

using namespace std;

int main() {
    std::cout << "Dichotomy method: " <<
              CompMethods::DichotomyMethod(CompMethods::default_func, -100.0, 100.0) << "\n";
    std::cout << "Newton method: " << CompMethods::NewtonMethod(CompMethods::default_func,
                                                                CompMethods::default_func_der, 10.0) << "\n";
    std::cout << "Relaxation method: " << CompMethods::RelaxationMethod(CompMethods::func_for_relaxation,
                                                                        0.5, 1.0) << "\n";
    return 0;
}
