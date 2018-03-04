#pragma once

#include <vector>
#include "sparse_matrix.h"

class Davidson {
 public:
  void diagonalize(const SparseMatrix&, const std::vector<double>&){};

  double get_eigenvalue() { return 0.0; }

  std::vector<double> get_eigenvector() { return std::vector<double>(); }
};
