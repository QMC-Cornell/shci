#pragma once

#include <climits>
#include <complex>
#include <vector>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "sparse_vector.h"

class SparseMatrix {
 public:
  double get_diag(const size_t i) const { return diag[i]; }
  
  std::complex<double> get_diag_green(const size_t i) const { return green_offset - diag[i]; }

  void cache_diag();

  std::vector<double> mul(const std::vector<double>& vec) const;

  // Multiply the Green's denominator. (w + n i) I - H
  std::vector<std::complex<double>> mul_green(const std::vector<std::complex<double>>& vec) const;

  void append_elem(const size_t i, const size_t j, const double& elem);

  void set_dim(const size_t dim);

  void set_green(const std::complex<double>& offset) { green_offset = offset; }

  void clear();

  void sort_row(const size_t i);

  void print_row(const size_t i) { rows[i].print(); }

  // transform each element to a * x + b.
  void transform(const double a, const double b);

 private:
  std::vector<SparseVector> rows;

  std::vector<double> diag_local;

  std::vector<double> diag;

  std::complex<double> green_offset;

  std::vector<double> reduce_sum(const std::vector<double>& vec) const;
};
