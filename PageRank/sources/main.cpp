#include <iostream>
#include <fstream>
#include "PageRank.h"
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

int main() {
    std::ifstream fin("graph.txt");
    auto graph = readGraphAdjacencyList(fin);
    auto matrix = graphToModifiedAdjacencyMatrix(graph);
    auto eigenVector = powerIteration(matrix, vector<double>(matrix.size1(), 1.0/matrix.size1()));
    //auto eigenVector = PageRank(graph);
    std::cout << "x: " << eigenVector << std::endl;
    vector<double> Ax = prod(matrix, eigenVector);
    Ax /= norm_1(Ax);
    std::cout << "Ax / ||Ax||1: " << Ax << std::endl;
    std::cout << "||Ax - x||1: " << norm_1(Ax - eigenVector) << std::endl;
    return 0;
}
