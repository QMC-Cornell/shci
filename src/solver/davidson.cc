#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>
#include "../config.h"

void Davidson::diagonalize(
    const SparseMatrix& matrix,
    const std::vector<double>& initial_vector,
    const double target_error,
    const bool verbose) {
  const double TOLERANCE = target_error;
  const size_t N_ITERATIONS_STORE = 5;

  const size_t dim = initial_vector.size();
  lowest_eigenvector.resize(dim);

  if (dim == 1) {
    lowest_eigenvalue = matrix.get_diag(0);
    lowest_eigenvector.resize(1);
    lowest_eigenvector[0] = 1.0;
    converged = true;
    return;
  }

  const size_t n_iterations_store = std::min(dim, N_ITERATIONS_STORE);
  double lowest_eigenvalue_prev = 0.0;

  std::vector<std::vector<double>> v(n_iterations_store);
  std::vector<std::vector<double>> Hv(n_iterations_store);
  std::vector<double> w(dim);
  std::vector<double> Hw(dim);
  for (size_t i = 0; i < n_iterations_store; i++) {
    v[i].resize(dim);
  }
  double norm = sqrt(Util::dot_omp(initial_vector, initial_vector));
#pragma omp parallel for
  for (size_t j = 0; j < dim; j++) v[0][j] = initial_vector[j] / norm;

  Eigen::MatrixXd h_krylov = Eigen::MatrixXd::Zero(n_iterations_store, n_iterations_store);
  std::vector<double> eigenvector_krylov(n_iterations_store);
  converged = false;
  Hv[0] = matrix.mul(v[0]);
  lowest_eigenvalue = Util::dot_omp(v[0], Hv[0]);
  h_krylov(0, 0) = lowest_eigenvalue;
  w = v[0];
  Hw = Hv[0];
  if (verbose) printf("Davidson #0: %.10f\n", lowest_eigenvalue);
  lowest_eigenvalue_prev = lowest_eigenvalue;

  size_t it_real = 1;
  for (size_t it = 1; it < n_iterations_store * 2; it++) {
    size_t it_circ = it % n_iterations_store;
    if (it >= n_iterations_store && it_circ == 0) {
      v[0] = w;
      Hv[0] = Hw;
      lowest_eigenvalue = Util::dot_omp(v[0], Hv[0]);
      h_krylov(0, 0) = lowest_eigenvalue;
      continue;
    }

#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      const double diff_to_diag = lowest_eigenvalue - matrix.get_diag(j);  // diag_elems[j];
      if (std::abs(diff_to_diag) < 1.0e-12) {
        v[it_circ][j] = (Hw[j] - lowest_eigenvalue * w[j]) / -1.0e-12;
      } else {
        v[it_circ][j] = (Hw[j] - lowest_eigenvalue * w[j]) / diff_to_diag;
      }
    }

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it_circ; i++) {
      norm = Util::dot_omp(v[it_circ], v[i]);
#pragma omp parallel for
      for (size_t j = 0; j < dim; j++) {
        v[it_circ][j] -= norm * v[i][j];
      }
    }
    norm = sqrt(Util::dot_omp(v[it_circ], v[it_circ]));
//  std::cout<<"\nnorm "<<norm<<"\n";

#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      v[it_circ][j] /= norm;
    }
    Hv[it_circ] = matrix.mul(v[it_circ]);

    if (norm<1e-12) {
      break;
//    for (size_t i = 0; i < it_circ + 1; i++) {
//      double norm = Util::dot_omp(v[i], v[i]);
//      for (size_t j = 0; j < dim; j++) v[i][j] /= norm;
//      for (size_t k = i + 1; k < it_circ + 1; k++) {
//        norm = Util::dot_omp(v[i], v[k]);
//        for (size_t j = 0; j < dim; j++) v[k][j] -= norm * v[i][j];
//      }
//    }
    }

    // Construct subspace matrix.
    for (size_t i = 0; i <= it_circ; i++) {
      h_krylov(i, it_circ) = Util::dot_omp(v[i], Hv[it_circ]);
      h_krylov(it_circ, i) = h_krylov(i, it_circ);
    }

    // Diagonalize subspace matrix.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
        h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1));
    lowest_eigenvalue = eigenSolver.eigenvalues()(0);
    const auto& eigenvectors = eigenSolver.eigenvectors();
    double factor = 1.0;
    if (eigenvectors(0, 0) < 0) factor = -1.0;
    for (size_t i = 0; i < it_circ + 1; i++) eigenvector_krylov[i] = eigenvectors(i, 0) * factor;
#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      double w_j = 0.0;
      double Hw_j = 0.0;
      for (size_t i = 0; i < it_circ + 1; i++) {
        w_j += v[i][j] * eigenvector_krylov[i];
        Hw_j += Hv[i][j] * eigenvector_krylov[i];
      }
      w[j] = w_j;
      Hw[j] = Hw_j;
    }

    if (verbose) printf("Davidson #%zu: %.10f\n", it_real, lowest_eigenvalue);
    it_real++;
    if (std::abs(lowest_eigenvalue - lowest_eigenvalue_prev) < TOLERANCE) {
      converged = true;
    } else {
      lowest_eigenvalue_prev = lowest_eigenvalue;
    }

    if (converged) break;
  }
  lowest_eigenvector = w;
}
