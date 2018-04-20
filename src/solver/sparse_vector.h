#pragma once

#include <cstddef>
#include <vector>

template <class T>
class SparseVector {
 public:
  void append(const size_t index, const T value) {
    indices.push_back(index);
    values.push_back(value);
  }

  size_t size() const { return indices.size(); }

  void get_index(const size_t i) const { return indices[i]; }

  void get_value(const size_t i) const { return values[i]; }

 private:
  std::vector<size_t> indices;

  std::vector<T> values;
};
