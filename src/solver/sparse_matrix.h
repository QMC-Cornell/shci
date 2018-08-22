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

  void cache_diag();

  std::vector<double> mul(const std::vector<double>& vec) const;

  std::vector<std::complex<double>> mul(const std::vector<std::complex<double>>& vec) const;

  void mul(
      const std::vector<double>& input_real,
      const std::vector<double>& input_imag,
      std::vector<double>& output_real,
      std::vector<double>& output_imag) const;

  void append_elem(const size_t i, const size_t j, const double& elem);

  void set_dim(const size_t dim);

  void clear();

  void sort_row(const size_t i);

  void print_row(const size_t i) { rows[i].print(); }

  std::vector<std::vector<size_t>> get_connections() const;

 private:
  std::vector<SparseVector> rows;
  
  std::vector<double> diag_local;

  std::vector<double> diag;

  std::vector<double> reduce_sum(const std::vector<double>& vec) const;
};
