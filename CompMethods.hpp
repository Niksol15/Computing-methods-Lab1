//
// Created by solon on 18.02.2021.
//

#ifndef _COMPMETHODS_H
#define _COMPMETHODS_H

#include <limits>
#include <type_traits>
#include <cmath>
#include <stdexcept>

namespace {
    constexpr char kWrongArgumentError[] = "Wrong argument at dichotomy method: The function at the ends of\n"
                                           "the segment must take values of different signs";
    constexpr char kWrongEpsError[] = "Epsilon must be greater then 0";
}

namespace CompMethods {
    constexpr double default_epsilon = 1e-6;

    constexpr double default_lambda = 1e-6;

    constexpr double func_for_relaxation(double x) {
        return sin(x) - x * x * x;
    }

    constexpr double default_func(double x) {
        return pow(x, 15) - 3 * x * sin(x) + pow(x, 4) - 10.0;
    }

    constexpr double default_func_der(double x) {
        return 4 * pow(x, 3) + 15 * pow(x, 14) - 3 * x * cos(x) - 3 * sin(x);
    }

    template<typename T>
    using IsFloatingPoint = std::enable_if_t<std::is_floating_point_v<T>, bool>;

    template<typename Function, typename FloatPoint>
    constexpr FloatPoint DichotomyMethod(Function &&func,
                                         FloatPoint leftEnd = std::numeric_limits<FloatPoint>::min(),
                                         FloatPoint rightEnd = std::numeric_limits<FloatPoint>::max(),
                                         FloatPoint eps = std::numeric_limits<FloatPoint>::epsilon(),
                                         IsFloatingPoint<FloatPoint> = true) {
        if (func(leftEnd) * func(rightEnd) > 0.0 || eps < 0) {
            throw std::logic_error(kWrongArgumentError);
        }
        while (rightEnd - leftEnd > eps) {
            FloatPoint middle = (leftEnd + rightEnd) / 2.0;
            if (func(middle) * func(rightEnd) < 0.0) {
                leftEnd = middle;
            } else {
                rightEnd = middle;
            }
        }
        return (leftEnd + rightEnd) / 2.0;
    }

    template<typename Function, typename Derivative, typename FloatPoint>
    constexpr FloatPoint NewtonMethod(Function &&func, Derivative &&dfunc,
                                      FloatPoint firstApproach,
                                      FloatPoint eps = std::numeric_limits<FloatPoint>::epsilon(),
                                      IsFloatingPoint<FloatPoint> = true) noexcept {
        eps = fabs(eps);
        FloatPoint curr, next = firstApproach;
        do {
            curr = next;
            next = curr - func(curr) / dfunc(curr);
        } while (fabs(next - curr) > eps);
        return next;
    }

    template<typename Function, typename FloatPoint>
    FloatPoint RelaxationMethod(Function &&func, FloatPoint lambda,
                                FloatPoint firstApproach,
                                FloatPoint eps = std::numeric_limits<FloatPoint>::epsilon(),
                                IsFloatingPoint<FloatPoint> = true) noexcept {
        eps = fabs(eps);
        FloatPoint curr, next = firstApproach;
        do {
            curr = next;
            next = curr + lambda * func(curr);
        } while (fabs(next - curr) > eps);
        return next;
    }
}


#endif //_COMPMETHODS_H
