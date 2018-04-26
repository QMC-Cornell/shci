#pragma once

#include <vector>
#include "sparse_matrix.h"

class Davidson {
 public:
  void diagonalize(
      const SparseMatrix<double>& matrix,
      const std::vector<double>& initial_vector,
      const bool verbose = false);

  double get_lowest_eigenvalue() { return lowest_eigenvalue; }

  std::vector<double> get_lowest_eigenvector() { return lowest_eigenvector; }

 private:
  double lowest_eigenvalue;

  std::vector<double> lowest_eigenvector;
};
