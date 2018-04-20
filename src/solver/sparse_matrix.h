#pragma once

#include "sparse_vector.h"

template <class T>
class SparseMatrix {
 public:
  double get_diag(const size_t i) const { return diag[i]; }

  std::vector<double> mul(const std::vector<double>& vec) const { return vec; }

 private:
  std::vector<SparseVector<T>> rows;

  std::vector<double> diag;
};

template <class T>
std::vector<double> SparseMatrix::mul(const std::vector<double>& vec) const {
  // TODO: Factor out raw parallel codes into a framework.
  const int proc_id = Parallel::get_proc_id();
  const int n_procs = Parallel::get_n_procs();
  const int n_threads = Parallel::get_n_threads();
  const size_t dim = rows.size();
  std::vector<std::vector<double>> res(n_threads);
  for (int i = 0; i < n_threads; i++) res[i].resize(dim, 0.0);
  std::vector<unsigned long long> n_nonzero_elems(n_threads, 0);

#pragma omp parallel for schedule(static, 1)
  for (size_t i = proc_id; i < dim; i += n_procs) {
    const int thread_id = omp_get_thread_num();
    const auto& row = rows[i];
    for (size_t j_id = 0; j_id < row.size(); j_id++) {
      const size_t j = row.get_index[j_id];
      const double H_ij = row.get_value[j_id];
      res[thread_id][i] += value * vec[j];
      if (i != j) {
        res[thread_id][j] += H_ij * vec[i];
        n_nonzero_elems[thread_id] += 2;
      } else {
        n_nonzero_elems[thread_id]++;
      }
    }
  }

  for (int i = 1; i < n_threads; i++) {
    n_nonzero_elems[0] += n_nonzero_elems[i];
  }

#pragma omp parallel for
  for (int j = 0; j < dim; j++) {
    for (int i = 1; i < n_threads; i++) {
      res[0][j] += res[i][j];
    }
  }

  

  // parallel->reduce_to_sum(res[0]);
  // parallel->reduce_to_sum(n_nonzero_elems[0]);


  if (verbose && first_iteration) {
    printf("Number of non-zero elements: %'llu\n", n_nonzero_elems[0]);
  }

  session->get_timer()->checkpoint("hamiltonian applied");

  return res[0];
}
