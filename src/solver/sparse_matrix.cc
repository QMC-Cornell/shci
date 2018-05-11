#include "sparse_matrix.h"

void SparseMatrix::append_elem(const size_t i, const size_t j, const double& elem) {
  rows[i].append(j, elem);
  if (i == j) diag[i] = elem;
}

std::vector<double> SparseMatrix::mul(const std::vector<double>& vec) const {
  // TODO: Factor out raw parallel codes into a framework.
  const int proc_id = Parallel::get_proc_id();
  const int n_procs = Parallel::get_n_procs();
  const int n_threads = Parallel::get_n_threads();
  const size_t dim = rows.size();
  std::vector<std::vector<double>> res_local(n_threads);
  std::vector<double> res(dim);
  for (int i = 0; i < n_threads; i++) res_local[i].assign(dim, 0.0);

#pragma omp parallel for schedule(static, 1)
  for (size_t i = proc_id; i < dim; i += n_procs) {
    const int thread_id = omp_get_thread_num();
    const auto& row = rows[i];
    for (size_t j_id = 0; j_id < row.size(); j_id++) {
      const size_t j = row.get_index(j_id);
      const double H_ij = row.get_value(j_id);
      res_local[thread_id][i] += H_ij * vec[j];
      if (i != j) res_local[thread_id][j] += H_ij * vec[i];
    }
  }

#pragma omp parallel for
  for (size_t j = 0; j < dim; j++) {
    for (int i = 1; i < n_threads; i++) {
      res_local[0][j] += res_local[i][j];
    }
  }

  MPI_Allreduce(res_local[0].data(), res.data(), dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return res;
}

void SparseMatrix::set_dim(const size_t dim) {
  rows.resize(dim);
  diag.resize(dim, 0.0);
}

void SparseMatrix::clear() { rows.clear(); }

void SparseMatrix::sort_row(const size_t i) { rows[i].sort(); }

void SparseMatrix::cache_diag() {
  std::vector<double> diag_local = diag;
  MPI_Allreduce(diag_local.data(), diag.data(), diag.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
