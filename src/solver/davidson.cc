#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>

void Davidson::diagonalize(
    const SparseMatrix<double>& matrix,
    const std::vector<double>& initial_vector,
    const bool verbose) {
  const double TOLERANCE = 1.0e-10;
  const double EPSILON = 1.0e-8;
  const size_t MAX_N_INTERATIONS = 12;

  const size_t dim = initial_vector.size();

  if (dim == 1) {
    lowest_eigenvalue = matrix.get_diag(0);
    lowest_eigenvector.resize(1);
    lowest_eigenvector[0] = 1.0;
    return;
  }

  const size_t max_n_iterations = std::min(dim, MAX_N_INTERATIONS);
  double lowest_eigenvalue_prev = 0.0;
  double residual_norm = 0.0;

  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(dim, max_n_iterations);
  for (size_t i = 0; i < dim; i++) v(i, 0) = initial_vector[i];
  v.col(0).normalize();

  Eigen::MatrixXd Hv = Eigen::MatrixXd::Zero(dim, max_n_iterations);
  Eigen::VectorXd w = Eigen::VectorXd::Zero(dim);
  Eigen::VectorXd Hw = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd h_krylov = Eigen::MatrixXd::Zero(max_n_iterations, max_n_iterations);
  Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(max_n_iterations);
  size_t len_work = 3 * max_n_iterations - 1;
  Eigen::VectorXd work(len_work);
  bool converged = false;
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

  for (size_t it = 1; it < max_n_iterations; it++) {
    // Compute residual.
    for (size_t j = 0; j < dim; j++) {
      const double diff_to_diag = lowest_eigenvalue - diag_elems[j];
      if (std::abs(diff_to_diag) < EPSILON) {
        v(j, it) = -1.0;
      } else {
        v(j, it) = (Hw(j, 0) - lowest_eigenvalue * w(j, 0)) / diff_to_diag;
      }
    }

    // If residual is small, converge.
    residual_norm = v.col(it).norm();
    if (residual_norm < TOLERANCE) converged = true;

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it; i++) {
      double norm = v.col(it).dot(v.col(i));
      v.col(it) -= norm * v.col(i);
    }
    v.col(it).normalize();

    // Apply H once.
    for (size_t i = 0; i < dim; i++) tmp_v[i] = v(i, it);
    const auto& tmp_Hv2 = matrix.mul(tmp_v);
    for (size_t i = 0; i < dim; i++) Hv(i, it) = tmp_Hv2[i];

    // Construct subspace matrix.
    for (size_t i = 0; i <= it; i++) {
      h_krylov(i, it) = v.col(i).dot(Hv.col(it));
      h_krylov(it, i) = h_krylov(i, it);
    }

    // Diagonalize subspace matrix.
    len_work = 3 * it + 2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
        h_krylov.leftCols(it + 1).topRows(it + 1));
    const auto& eigenvalues = eigenSolver.eigenvalues();
    auto eigenvectors = eigenSolver.eigenvectors();
    if (eigenvectors(0, 0) < 0) eigenvectors = -eigenvectors;
    lowest_eigenvalue = eigenvalues[0];
    size_t lowest_id = 0;
    for (size_t i = 1; i <= it; i++) {
      if (eigenvalues[i] < lowest_eigenvalue) {
        lowest_eigenvalue = eigenvalues[i];
        lowest_id = i;
      }
    }
    w = v.leftCols(it) * eigenvectors.col(lowest_id).topRows(it);
    Hw = Hv.leftCols(it) * eigenvectors.col(lowest_id).topRows(it);

    if (verbose) printf("Davidson #%zu: %.10f\n", it, lowest_eigenvalue);
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
