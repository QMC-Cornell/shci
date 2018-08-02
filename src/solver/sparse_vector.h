#pragma once

#include <cassert>
#include <cstddef>
#include <vector>
#include "../util.h"

class SparseVector {
 public:
  void append(const size_t index, const double value) {
    indices.push_back(index);
    values.push_back(value);
  }

  size_t size() const { return indices.size(); }

  size_t get_index(const size_t i) const { return indices.at(i); }

  double get_value(const size_t i) const { return values.at(i); }

  void sort() {
    Util::sort_by_first<size_t, double>(indices, values);
    for (size_t i = 1; i < size(); i++) assert(indices[i] > indices[i - 1]);
  }

  void print() const {
    for (size_t i = 0; i < size(); i++) printf("%zu: %.12f\n", indices[i], values[i]);
    printf("n elems: %zu\n", size());
  }

  // transform each element to a * x + b.
  void transform(const double a, const double b) {
    const size_t n = size();
    for (size_t i = 0; i < n; i++) {
      values[i] = values[i] * a + b;
    }
  }

 private:
  std::vector<size_t> indices;

  std::vector<double> values;
};
