//
// Created by niksol on 18.03.2021.
//

#include "PageRank.h"
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <sstream>

namespace ublas = boost::numeric::ublas;

static constexpr size_t kIterationLimit = 10000;

ublas::vector<double> powerIteration(const ublas::matrix<double> &matrix,
                                     const ublas::vector<double> &firstApproach, double eps) {
    ublas::vector<double> next = firstApproach / ublas::norm_1(firstApproach);
    ublas::vector<double> curr(next.size(), 0);
    for (size_t i = 1; i < kIterationLimit && norm_2(next - curr) > eps; ++i) {
        std::cout << next << std::endl;
        curr = next;
        next = prod(matrix, curr);
        next /= norm_1(next);


    }
    return next;
}

std::vector<std::set<size_t>> readGraphAdjacencyList(std::istream &is) {
    size_t n;
    is >> n;
    is.get();
    std::vector<std::set<size_t>> res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::string currString;
        getline(is, currString);
        if (currString.empty()) {
            throw std::logic_error("There should be no dead-end vertices in the graph");
        }
        std::istringstream iss(currString);
        while (iss) {
            size_t vertex;
            iss >> vertex;
            if (vertex >= n) {
                throw std::logic_error("Out of range in vertex " + std::to_string(i) +
                                       ": references to " + std::to_string(vertex));
            }
            if (vertex != i) {
                res[i].insert(vertex);
            }
            iss.get();
        }
    }
    return res;
}

ublas::matrix<double> graphToModifiedAdjacencyMatrix(
        const std::vector<std::set<size_t>> &adjacencyList) {
    ublas::matrix<double> res(adjacencyList.size(), adjacencyList.size());
    for (size_t i = 0; i < adjacencyList.size(); ++i) {
        double val = 1.0 / adjacencyList[i].size();
        for (size_t vertex: adjacencyList[i]) {
            if (vertex >= adjacencyList.size()) {
                throw std::logic_error("Wrong graph");
            }
            res(vertex, i) = val;
        }
    }
    return res;
}

ublas::vector<double> PageRank(const std::vector<std::set<size_t>> &adjacencyList, double eps) {
    auto matrix = graphToModifiedAdjacencyMatrix(adjacencyList);
    if (matrix.size1() != matrix.size2()) {
        throw std::logic_error("Matrix must be square");
    }
    return powerIteration(matrix, ublas::vector<double>(matrix.size1(), 1.0 / matrix.size1()));
}
