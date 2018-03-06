#pragma once

#include <vector>

template <class T>
class SparseVector {
 public:
  std::vector<size_t> indices;

  std::vector<T> values;

  void append(const size_t index, const T value) {
    indices.push_back(index);
    values.push_back(value);
  }
};
