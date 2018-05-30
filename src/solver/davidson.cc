#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>

void Davidson::diagonalize(
    const SparseMatrix& matrix,
    const std::vector<double>& initial_vector,
    const bool verbose,
    const bool until_converged) {
  const double TOLERANCE = until_converged ? 2.0e-7 : 2.0e-6;
  const size_t N_ITERATIONS_STORE = 4;

  const size_t dim = initial_vector.size();

  if (dim == 1) {
    lowest_eigenvalue = matrix.get_diag(0);
    lowest_eigenvector.resize(1);
    lowest_eigenvector[0] = 1.0;
    converged = true;
    return;
  }

  const size_t n_iterations_store = std::min(dim, N_ITERATIONS_STORE);
  double lowest_eigenvalue_prev = 0.0;

  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(dim, n_iterations_store);
  for (size_t i = 0; i < dim; i++) v(i, 0) = initial_vector[i];
  v.col(0).normalize();

  Eigen::MatrixXd Hv = Eigen::MatrixXd::Zero(dim, n_iterations_store);
  Eigen::VectorXd w = Eigen::VectorXd::Zero(dim);
  Eigen::VectorXd Hw = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd h_krylov = Eigen::MatrixXd::Zero(n_iterations_store, n_iterations_store);
  Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(n_iterations_store);
  size_t len_work = 3 * n_iterations_store - 1;
  Eigen::VectorXd work(len_work);
  converged = false;
  std::vector<double> tmp_v(dim);
  // Get diagonal elements.
  Eigen::VectorXd diag_elems(dim);
  for (size_t i = 0; i < dim; i++) diag_elems[i] = matrix.get_diag(i);

  // First iteration.
  for (size_t i = 0; i < dim; i++) tmp_v[i] = v(i, 0);
  const auto& tmp_Hv = matrix.mul(tmp_v);
  for (size_t i = 0; i < dim; i++) Hv(i, 0) = tmp_Hv[i];
  lowest_eigenvalue = v.col(0).dot(Hv.col(0));
  h_krylov(0, 0) = lowest_eigenvalue;
  w = v.col(0);
  Hw = Hv.col(0);
  if (verbose) printf("Davidson #0: %.10f\n", lowest_eigenvalue);

  size_t it_real = 1;
  for (size_t it = 1; it < n_iterations_store * 3; it++) {
    size_t it_circ = it % n_iterations_store;
    if (it >= n_iterations_store && it_circ == 0) {
      v.col(0) = w.col(0);
      Hv.col(0) = Hw.col(0);
      lowest_eigenvalue = v.col(0).dot(Hv.col(0));
      h_krylov(0, 0) = lowest_eigenvalue;
      continue;
    }

    for (size_t j = 0; j < dim; j++) {
      const double diff_to_diag = lowest_eigenvalue - diag_elems[j];
      if (std::abs(diff_to_diag) < 1.0e-12) {
        v(j, it_circ) = (Hw(j, 0) - lowest_eigenvalue * w(j, 0)) / -1.0e-12;
      } else {
        v(j, it_circ) = (Hw(j, 0) - lowest_eigenvalue * w(j, 0)) / diff_to_diag;
      }
    }

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it_circ; i++) {
      double norm = v.col(it_circ).dot(v.col(i));
      v.col(it_circ) -= norm * v.col(i);
    }
    v.col(it_circ).normalize();

    // Apply H once.
    for (size_t i = 0; i < dim; i++) tmp_v[i] = v(i, it_circ);
    const auto& tmp_Hv2 = matrix.mul(tmp_v);
    for (size_t i = 0; i < dim; i++) Hv(i, it_circ) = tmp_Hv2[i];

    // Construct subspace matrix.
    for (size_t i = 0; i <= it_circ; i++) {
      h_krylov(i, it_circ) = v.col(i).dot(Hv.col(it_circ));
      h_krylov(it_circ, i) = h_krylov(i, it_circ);
    }

    // Diagonalize subspace matrix.
    len_work = 3 * it_circ + 2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
        h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1));
    const auto& eigenvalues = eigenSolver.eigenvalues();
    auto eigenvectors = eigenSolver.eigenvectors();
    lowest_eigenvalue = eigenvalues(0);
    if (eigenvectors(0, 0) < 0) eigenvectors.col(0) *= -1;
    w = v.leftCols(it_circ + 1) * eigenvectors.col(0).topRows(it_circ + 1);
    Hw = Hv.leftCols(it_circ + 1) * eigenvectors.col(0).topRows(it_circ + 1);

    if (verbose) printf("Davidson #%zu: %.10f\n", it_real, lowest_eigenvalue);
    it_real++;
    if (std::abs(lowest_eigenvalue - lowest_eigenvalue_prev) < TOLERANCE) {
      converged = true;
    } else {
      lowest_eigenvalue_prev = lowest_eigenvalue;
    }

    if (converged) break;
  }

  lowest_eigenvector.resize(dim);
  for (unsigned i = 0; i < dim; i++) lowest_eigenvector[i] = w(i);
}
