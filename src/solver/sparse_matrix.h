#pragma once

#include "sparse_vector.h"

template <class T>
class SparseMatrix {
 public:
  std::vector<SparseVector<T>> rows;
};
