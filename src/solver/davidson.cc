#include "davidson.h"

void Davidson::diagonalize(const SparseMatrix<double>&, const std::vector<double>& initial) {
  eigenvector = initial;
}
