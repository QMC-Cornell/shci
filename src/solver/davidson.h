#pragma once

#include <vector>
#include "sparse_matrix.h"

class Davidson {
 public:
  void diagonalize(
      const SparseMatrix& matrix,
      const std::vector<double>& initial_vector,
      const bool verbose = false,
      const bool until_converged = true);

  double get_lowest_eigenvalue() const { return lowest_eigenvalue; }

  std::vector<double> get_lowest_eigenvector() const { return lowest_eigenvector; }

  bool converged;

 private:
  double lowest_eigenvalue;

  std::vector<double> lowest_eigenvector;
};
