#pragma once

#include "sparse_vector.h"

template <class T>
class SparseMatrix {
 public:
  double get_diag(const size_t i) const { return rows[i][0]; }

  std::vector<double> mul(const std::vector<double>& vec) const;

 private:
  std::vector<SparseVector<T>> rows;
};
