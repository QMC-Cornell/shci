#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>
#include "../config.h"
#include <random>

void Davidson::diagonalize(
    const SparseMatrix& matrix,
    const std::vector<std::vector<double>>& initial_vectors,
    const double target_error,
    const bool verbose) {
  const double TOLERANCE = target_error;
  const size_t N_ITERATIONS_STORE = 100;

  const size_t dim = initial_vectors[0].size();
  const unsigned n_states = std::min(dim, initial_vectors.size());
  for (auto& eigenvec : lowest_eigenvectors) eigenvec.resize(dim);

  if (dim == 1) {
    lowest_eigenvalues[0] = matrix.get_diag(0);
    lowest_eigenvectors[0].resize(1);
    lowest_eigenvectors[0][0] = 1.0;
    converged = true;
    return;
  }

  const size_t n_iterations_store = std::min(dim, N_ITERATIONS_STORE);
  std::vector<double> lowest_eigenvalues_prev(n_states, 0.0);

  std::vector<std::vector<double>> v(n_states * n_iterations_store);
  std::vector<std::vector<double>> Hv(n_states * n_iterations_store);
  std::vector<std::vector<double>> w(n_states);
  std::vector<std::vector<double>> Hw(n_states);
  for (size_t i = 0; i < v.size(); i++) {
    v[i].resize(dim);
  }

  for (size_t i = 0; i < n_states; i++) {
    w[i].resize(dim);
    Hw[i].resize(dim);
  }

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    double norm = sqrt(Util::dot_omp(initial_vectors[i_state], initial_vectors[i_state]));
//#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) v[i_state][j] = initial_vectors[i_state][j] / norm;
    if (i_state > 0) {
      // Orthogonalize
      double inner_prod;
      for (unsigned k_state = 0; k_state < i_state; k_state++) {
        inner_prod = Util::dot_omp(v[i_state], v[k_state]);
        for (size_t j = 0; j < dim; j++) v[i_state][j] -= inner_prod * v[k_state][j];
      }
      // Normalize
      norm = sqrt(Util::dot_omp(v[i_state], v[i_state]));
      for (size_t j = 0; j < dim; j++) v[i_state][j] /= norm;
    }
  }

  Eigen::MatrixXd h_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  Eigen::MatrixXd eigenvector_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  converged = false;

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    Hv[i_state] = matrix.mul(v[i_state]);
    lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
    h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
    w[i_state] = v[i_state];
    Hw[i_state] = Hv[i_state];
  }
  if (verbose) {
    printf("Davidson #0:");
    for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
    printf("\n");
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0., 1.);

  size_t it_real = 1;
  for (size_t it = n_states; it < n_iterations_store * 10; it++) {
    size_t it_circ = it % n_iterations_store;
//std::cout<<"\nit_real it_circ it "<<it_real<<" "<<it_circ<<" "<<it<<std::endl;
    if (it >= n_iterations_store && it_circ == 0) {
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        v[i_state] = w[i_state];
        Hv[i_state] = Hw[i_state];
      }
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
        h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
        for (unsigned k_state = i_state + 1; k_state < n_states; k_state++) {
          double element = Util::dot_omp(v[i_state], Hv[k_state]);
          h_krylov(i_state, k_state) = element;
          h_krylov(k_state, i_state) = element;
        }
      }
      continue;
    }
    size_t i = it_circ % n_states;
//#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      const double diff_to_diag = lowest_eigenvalues[i] - matrix.get_diag(j);  // diag_elems[j];
      if (std::abs(diff_to_diag) < 1.0e-8) {
std::cout<<"\ndiff_to_diag <1e-12 = "<< diff_to_diag <<std::endl; 
        //v[it_circ][j] = (Hw[i][j] - lowest_eigenvalues[i] * w[i][j]) / -1.0e-12;
        //v[it_circ][j] = distribution(generator);
        v[it_circ][j] = 0.;
      } else {
        v[it_circ][j] = (Hw[i][j] - lowest_eigenvalues[i] * w[i][j]) / diff_to_diag;
      }
    }
//std::cout<<"\nv[it_circ]:";
//for (const auto& val: v[it_circ]) std::cout<<" "<<val;	    
//std::cout<<"\n";

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it_circ; i++) {
      double norm = Util::dot_omp(v[it_circ], v[i]);
//#pragma omp parallel for
      for (size_t j = 0; j < dim; j++) {
        v[it_circ][j] -= norm * v[i][j];
      }
    }
//std::cout<<"\nv[it_circ] after orthogonalization:";
//for (const auto& val: v[it_circ]) std::cout<<" "<<val;	    
//std::cout<<"\n";
    double norm = sqrt(Util::dot_omp(v[it_circ], v[it_circ]));
//norm = std::max(norm, 1e-6);
//#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      v[it_circ][j] /= norm;
    }

if (norm < 1e-10) {
std::cout<<"\nnorm "<<norm<<" it_circ "<<it_circ<<std::endl;
//break;
// Reorthogonalization: MGS 
for (size_t i = 0; i < it_circ + 1; i++) {
  norm = Util::dot_omp(v[i], v[i]);
  for (size_t j = 0; j < dim; j++) v[i][j] /= norm;
  for (size_t k = i + 1; k < it_circ + 1; k++) {
    norm = Util::dot_omp(v[i], v[k]);
    for (size_t j = 0; j < dim; j++) v[k][j] -= norm * v[i][j];
  }
}
}
    Hv[it_circ] = matrix.mul(v[it_circ]);

    // Construct subspace matrix.
    for (size_t i = 0; i <= it_circ; i++) {
//std::cout<<"\nv["<<i<<"]:";
//for (const auto& val: v[i]) std::cout<<" "<<val;	    
      //h_krylov(i, it_circ) = Util::dot_omp(v[i], Hv[it_circ]);
      h_krylov(it_circ, i) = Util::dot_omp(v[i], Hv[it_circ]);
      //h_krylov(it_circ, i) = h_krylov(i, it_circ); // only lower trianguluar part is referenced
    }

//std::cout<<"\nbefore diagonalize\n";
//std::cout<<h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1)<<std::endl;    
    // Diagonalize subspace matrix.
    if ((it_circ + 1) % n_states == 0) {
    //if (true) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
          h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1));
      const auto& eigenvals = eigenSolver.eigenvalues();  // in ascending order
      const auto& eigenvecs = eigenSolver.eigenvectors();
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        lowest_eigenvalues[i_state] = eigenvals(i_state);
        double factor = 1.0;
        if (eigenvecs(0, i_state) < 0) factor = -1.0;
        for (size_t i = 0; i < it_circ + 1; i++)
          eigenvector_krylov(i, i_state) = eigenvecs(i, i_state) * factor;
//#pragma omp parallel for
        for (size_t j = 0; j < dim; j++) {
          double w_j = 0.0;
          double Hw_j = 0.0;
          for (size_t i = 0; i < it_circ + 1; i++) {
            w_j += v[i][j] * eigenvector_krylov(i, i_state);
            Hw_j += Hv[i][j] * eigenvector_krylov(i, i_state);
          }
          w[i_state][j] = w_j;
          Hw[i_state][j] = Hw_j;
        }
      }

      if (verbose) {
        printf("Davidson #%zu:", it_real);
        for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
        printf("\n");
      }
      it_real++;
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        if (std::abs(lowest_eigenvalues[i_state] - lowest_eigenvalues_prev[i_state]) > TOLERANCE) {
//std::cout<<"\n tolerance not satisfied"<<std::endl;
          break;
	}
        if (i_state == n_states - 1) converged = true;
      }
      if (!converged) lowest_eigenvalues_prev = lowest_eigenvalues;

      if (converged) break;
    }
  }
  lowest_eigenvectors = w;
  //std::exit(0);
}
