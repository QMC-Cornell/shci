#pragma once

#include <climits>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "sparse_vector.h"

template <class T>
class SparseMatrix {
 public:
  T get_diag(const size_t i) const { return rows[i].get_value(0); }

  std::vector<T> mul(const std::vector<T>& vec) const;

  void append_elem(const size_t i, const size_t j, const T& elem);

  void set_dim(const size_t dim);

  void clear();

  void sort_row(const size_t i);

#ifndef DEBUG
 private:
#endif
  std::vector<SparseVector<T>> rows;
};

template <class T>
void SparseMatrix<T>::append_elem(const size_t i, const size_t j, const T& elem) {
  rows[i].append(j, elem);
}

template <class T>
std::vector<T> SparseMatrix<T>::mul(const std::vector<T>& vec) const {
  // TODO: Factor out raw parallel codes into a framework.
  const int proc_id = Parallel::get_proc_id();
  const int n_procs = Parallel::get_n_procs();
  const int n_threads = Parallel::get_n_threads();
  const size_t dim = rows.size();
  std::vector<std::vector<T>> res(n_threads);
  for (int i = 0; i < n_threads; i++) res[i].resize(dim, 0.0);

#pragma omp parallel for schedule(static, 1)
  for (size_t i = proc_id; i < dim; i += n_procs) {
    const int thread_id = omp_get_thread_num();
    const auto& row = rows[i];
    for (size_t j_id = 0; j_id < row.size(); j_id++) {
      const size_t j = row.get_index(j_id);
      const T H_ij = row.get_value(j_id);
      res[thread_id][i] += H_ij * vec[j];
      if (i != j) res[thread_id][j] += H_ij * vec[i];
    }
  }

#pragma omp parallel for
  for (size_t j = 0; j < dim; j++) {
    for (int i = 1; i < n_threads; i++) {
      res[0][j] += res[i][j];
    }
  }

  if (dim <= 0 || dim >= INT_MAX) throw std::runtime_error("invalid number of dets");
  MPI_Allreduce(res[0].data(), res[1].data(), dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return res[1];
}

template <class T>
void SparseMatrix<T>::set_dim(const size_t dim) {
  rows.resize(dim);
}

template <class T>
void SparseMatrix<T>::clear() {
  rows.clear();
}

template <class T>
void SparseMatrix<T>::sort_row(const size_t i) {
 rows[i].sort();
}