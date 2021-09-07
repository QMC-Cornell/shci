#include "sparse_matrix.h"

#include <iostream>
#include <fgpl/src/dist_range.h>

#include "../util.h"

void SparseMatrix::append_elem(const size_t i, const size_t j, const double& elem) {
  rows[i].append(j, elem);
  if (i == j) diag_local[i] = elem;
}

size_t SparseMatrix::count_n_elems() const {
  // TODO: Factor out raw parallel codes into a framework.
  const int proc_id = Parallel::get_proc_id();
  const int n_procs = Parallel::get_n_procs();
  unsigned long long n_elems_local = 0;
  unsigned long long n_elems = 0;
  const size_t dim = rows.size();
  for (size_t i = proc_id; i < dim; i += n_procs) {
    n_elems_local += rows[i].size();
  }
  MPI_Allreduce(&n_elems_local, &n_elems, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  n_elems = n_elems * 2 - dim;
  return n_elems;  
}

std::vector<double> SparseMatrix::mul(const std::vector<double>& vec) const {
  // TODO: Factor out raw parallel codes into a framework.
  const int proc_id = Parallel::get_proc_id();
  const int n_procs = Parallel::get_n_procs();
  const size_t dim = rows.size();
  std::vector<double> res_local(dim, 0.0);

#pragma omp parallel for schedule(static, 1)
  for (size_t i = proc_id; i < dim; i += n_procs) {
    const auto& row = rows[i];
    double diff_i = 0.0;
    for (size_t j_id = 0; j_id < row.size(); j_id++) {
      const size_t j = row.get_index(j_id);
      const double H_ij = row.get_value(j_id);
      diff_i += H_ij * vec[j];
      if (i != j) {
        const double diff_j = H_ij * vec[i];
#pragma omp atomic
        res_local[j] += diff_j;
      }
    }
#pragma omp atomic
    res_local[i] += diff_i;
  }

  const auto& res = reduce_sum(res_local);

  return res;
}

std::vector<std::complex<double>> SparseMatrix::mul(
    const std::vector<std::complex<double>>& vec) const {
  const size_t dim = rows.size();
  std::vector<std::complex<double>> res(dim);
  std::vector<double> tmp(dim);
  std::vector<double> vec_tmp(dim);

  // First calculate the real part.
  for (size_t i = 0; i < dim; i++) {
    vec_tmp[i] = vec[i].real();
  }
  tmp = mul(vec_tmp);
  for (size_t i = 0; i < dim; i++) {
    res[i] = tmp[i];
  }

  // Calculate the imag part.
  for (size_t i = 0; i < dim; i++) {
    vec_tmp[i] = vec[i].imag();
  }
  tmp = mul(vec_tmp);
#pragma omp parallel for
  for (size_t i = 0; i < dim; i++) {
    res[i] += tmp[i] * Util::I;
  }

  return res;
}

void SparseMatrix::set_dim(const size_t dim) {
  rows.resize(dim);
  diag_local.resize(dim, 0.0);
  diag.resize(dim, 0.0);
}

void SparseMatrix::clear() {
  rows.clear();
  rows.shrink_to_fit();
  diag_local.clear();
  diag_local.shrink_to_fit();
  diag.clear();
  diag.shrink_to_fit();
}

void SparseMatrix::sort_row(const size_t i) { rows[i].sort(); }

void SparseMatrix::cache_diag() { diag = reduce_sum(diag_local); }

std::vector<double> SparseMatrix::reduce_sum(const std::vector<double>& vec) const {
  const size_t dim = vec.size();
  std::vector<double> res(dim, 0.0);
  const size_t TRUNK_SIZE = 1 << 27;
  double* src_ptr = const_cast<double*>(vec.data());
  double* dest_ptr = res.data();
  size_t n_elems_left = dim;
  while (n_elems_left > TRUNK_SIZE) {
    MPI_Allreduce(src_ptr, dest_ptr, TRUNK_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    n_elems_left -= TRUNK_SIZE;
    src_ptr += TRUNK_SIZE;
    dest_ptr += TRUNK_SIZE;
  }
  MPI_Allreduce(src_ptr, dest_ptr, n_elems_left, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return res;
}

std::vector<std::vector<size_t>> SparseMatrix::get_connections() const {
  std::vector<std::vector<size_t>> connections;
  for (const auto& row : rows) {
    connections.push_back(row.get_connections());
  }
  return connections;
}

void SparseMatrix::update_existing_elems(std::function<double(size_t, size_t, int)> get_hamiltonian_elem) {
  for (size_t k = 0; k < 5; k++) {
    fgpl::DistRange<size_t>(k, rows.size(), 5).for_each([&](const size_t i) {
      for (size_t j=0; j < rows[i].size(); j++) {
        rows[i].set_value(j, get_hamiltonian_elem(i, rows[i].get_index(j), -1));
      }
    });
    if (Parallel::is_master()) printf("%zu%% ", (k + 1) * 20);
  }
  if (Parallel::is_master()) printf("\n");
}
