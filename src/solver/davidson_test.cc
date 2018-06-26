#include "davidson.h"
#include <gtest/gtest.h>

// Test with a Hilbert matrix.
class HilbertSystem {
 public:
  SparseMatrix matrix;

  HilbertSystem(int n) {
    matrix.set_dim(n);
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        matrix.append_elem(i, j, get_hamiltonian(i, j));
      }
    }
    matrix.cache_diag();
  }

  double get_hamiltonian(int i, int j) {
    const double GAMMA = 10.0;
    if (i == j) return -1.0 / (2 * i + 1);
    return -1.0 / GAMMA / (i + j + 1);
  }
};

TEST(DavidsonTest, HilbertSystem) {
  const int N = 1000;
  HilbertSystem hilbert_system(N);
  Davidson davidson;

  const std::vector<double> expected_eigenvalues(
      {-1.00956719, -0.3518051, -0.23097854, -0.17336724, -0.13218651});
  const std::vector<std::vector<double>> expected_eigenvectors(
      {{0.99292536, 0.08026708, 0.04720676, 0.03412438, 0.02694173},
       {0.10953429, -0.90014126, -0.22310872, -0.14701356, -0.11467862},
       {0.04208261, 0.42251014, -0.54880665, -0.25894711, -0.19182954},
       {0.00259482, -0.02195869, -0.78985725, 0.1487066, 0.10266289},
       {0.01203533, 0.04023094, 0.09953056, -0.90203616, -0.06584302}});

  // Check eigenvalue and eigenvector with reference values from exact diagonalization.
  std::vector<double> initial_vector(N, 0.0);
  initial_vector[0] = 1.0;
  davidson.diagonalize(hilbert_system.matrix, initial_vector, 1.0e-6, true);
  const double lowest_eigenvalue = davidson.get_lowest_eigenvalue();
  EXPECT_NEAR(lowest_eigenvalue, expected_eigenvalues[0], 1.0e-6);
  const std::vector<double> lowest_eigenvector = davidson.get_lowest_eigenvector();
  for (int i = 0; i < 5; i++) {
    EXPECT_NEAR(lowest_eigenvector[i], expected_eigenvectors[0][i], 1.0e-4);
  }
}
