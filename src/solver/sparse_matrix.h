#pragma once

#include "sparse_vector.h"

template <class T>
class SparseMatrix {
 public:
  double get_diag(const size_t i) const { return diag[i]; }

  std::vector<double> get_diag() const { return diag; }

  std::vector<double> mul(const std::vector<double>& vec) const { return vec; }

 private:
  std::vector<SparseVector<T>> rows;

  std::vector<double> diag;
};
