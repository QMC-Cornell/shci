#pragma once

#include <vector>
#include "sparse_matrix.h"

class Davidson {
 public:
  Davidson(const unsigned n_states) {
    lowest_eigenvalues.resize(n_states);
    lowest_eigenvectors.resize(n_states);
  }

  void diagonalize(
      const SparseMatrix& matrix,
      const std::vector<std::vector<double>>& initial_vectors,
      const double target_error,
      const bool verbose = false);

  std::vector<double> get_lowest_eigenvalues() const { return lowest_eigenvalues; }

  std::vector<std::vector<double>> get_lowest_eigenvectors() const { return lowest_eigenvectors; }

  bool converged = false;

 private:
  std::vector<double> lowest_eigenvalues;

  std::vector<std::vector<double>> lowest_eigenvectors;
};
