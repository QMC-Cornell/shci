#pragma once

#include <vector>
#include "sparse_matrix.h"

class Davidson {
 public:
  void diagonalize(const SparseMatrix<double>& matrix, const std::vector<double>& initial);

  double get_eigenvalue() { return eigenvalue; }

  std::vector<double> get_eigenvector() { return eigenvector; }

 private:
  double eigenvalue;

  std::vector<double> eigenvector;
};
