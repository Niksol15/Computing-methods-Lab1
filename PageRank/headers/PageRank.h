//
// Created by niksol on 18.03.2021.
//

#ifndef PAGERANK_PAGERANK_H
#define PAGERANK_PAGERANK_H

#include <iostream>
#include <numeric>
#include <vector>
#include <set>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

boost::numeric::ublas::vector<double> powerIteration(
        const boost::numeric::ublas::matrix<double> &matrix,
        const boost::numeric::ublas::vector<double> &firstApproach,
        double eps = std::numeric_limits<double>::epsilon());

std::vector<std::set<size_t>> readGraphAdjacencyList(std::istream &is = std::cin);

boost::numeric::ublas::matrix<double> graphToModifiedAdjacencyMatrix(
        const std::vector<std::set<size_t>>& adjacencyList);

boost::numeric::ublas::vector<double> PageRank(const std::vector<std::set<size_t>>& adjacencyList,
                                               double eps = std::numeric_limits<double>::epsilon());

#endif //PAGERANK_PAGERANK_H
