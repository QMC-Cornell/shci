#pragma once

class DavidsonSolver {

public:
  DavidsonSolver(const SparseMatrix& hamiltonian_matrix,
           const Matrix<float, Dynamic, Dynamic, RowMajor>& hessian_ci_orb, 
           const Matrix<double, Dynamic, Dynamic, RowMajor>& hessian_orb_orb,
           const double energy_var):
  e_var(energy_var),
  Hamiltonian(hamiltonian_matrix),
  Hco(hessian_ci_orb),
  Hoo(hessian_orb_orb) {
    n_dets = hamiltonian_matrix.count_n_rows();
    n_orb_param = Hco.cols();
//std::cout<<"\nhamiltonian\n";
//for (size_t i = 0; i<n_dets; i++) hamiltonian_matrix.print_row(i);
//std::cout<<"\ne_var "<<e_var;
//std::cout<<"\nHco\n"<<Hco;
//std::cout<<"\nHoo\n"<<Hoo;
  }

  double lowest_eigenvalue;

  //std::vector<double> solve(const std::vector<double>& wf_coefs) const {
  std::vector<double> solve(const std::vector<double>& wf_coefs) {
    Timer::start("Davidson diagonalization solver");

    const double TOLERANCE = 1e-6;
    const size_t n_iterations_store = 5;
    double lowest_eigenvalue_prev = 0.;
 
    std::vector<double> initial_vector_c(wf_coefs), initial_vector_o(n_orb_param, 0.);
    for (size_t i = 0; i < n_orb_param; i++) initial_vector_o[i] = Hoo(i, i) > 1e-5 ? Hco(0, i) / Hoo(i, i) : 0.;

    std::vector<std::vector<double>> v_c(n_iterations_store), v_o(n_iterations_store);
    std::vector<std::vector<double>> Hv_c(n_iterations_store), Hv_o(n_iterations_store);
    std::vector<double> w_c(n_dets), w_o(n_orb_param);
    std::vector<double> Hw_c(n_dets), Hw_o(n_orb_param);
    for (size_t i = 0; i < n_iterations_store; i++) {
      v_c[i].resize(n_dets); v_o[i].resize(n_orb_param);
      Hv_c[i].resize(n_dets); Hv_o[i].resize(n_orb_param);
    }
    double norm = sqrt(Util::dot_omp(initial_vector_c, initial_vector_c) + Util::dot_omp(initial_vector_o, initial_vector_o));
#pragma omp parallel for
    for (size_t j = 0; j < n_dets; j++) v_c[0][j] = initial_vector_c[j] / norm;
#pragma omp parallel for
    for (size_t j = 0; j < n_orb_param; j++) v_o[0][j] = initial_vector_o[j] / norm;
  
    Eigen::MatrixXd h_krylov = Eigen::MatrixXd::Zero(n_iterations_store, n_iterations_store);
    std::vector<double> eigenvector_krylov(n_iterations_store);
    bool converged = false;
    mat_vec(v_c[0], v_o[0], Hv_c[0], Hv_o[0]);
    lowest_eigenvalue = Util::dot_omp(v_c[0], Hv_c[0]) + Util::dot_omp(v_o[0], Hv_o[0]);
    h_krylov(0, 0) = lowest_eigenvalue;
    w_c = v_c[0]; w_o = v_o[0];
    Hw_c = Hv_c[0]; Hw_o = Hv_o[0];
    printf("Davidson #0: %.10f\n", lowest_eigenvalue);
    lowest_eigenvalue_prev = lowest_eigenvalue;
  
    size_t it_real = 1;
    for (size_t it = 1; it < 2000; it++) {
      size_t it_circ = it % n_iterations_store;
      if (it >= n_iterations_store && it_circ == 0) {
        v_c[0] = w_c; v_o[0] = w_o;
        Hv_c[0] = Hw_c; Hv_o[0] = Hw_o;
        lowest_eigenvalue = Util::dot_omp(v_c[0], Hv_c[0]) + Util::dot_omp(v_o[0], Hv_o[0]);
        h_krylov(0, 0) = lowest_eigenvalue;
        continue;
      }
  
      const double diff_to_diag = lowest_eigenvalue; 
      if (std::abs(diff_to_diag) < 1.0e-12) {
        v_c[it_circ][0] = (Hw_c[0] - lowest_eigenvalue * w_c[0]) / -1.0e-12;
      } else {
        v_c[it_circ][0] = (Hw_c[0] - lowest_eigenvalue * w_c[0]) / diff_to_diag;
      }
#pragma omp parallel for
      for (size_t j = 1; j < n_dets; j++) {
        const double diff_to_diag = lowest_eigenvalue - 2 * (Hamiltonian.get_diag(j) - e_var);  // diag_elems[j];
        if (std::abs(diff_to_diag) < 1.0e-12) {
          v_c[it_circ][j] = (Hw_c[j] - lowest_eigenvalue * w_c[j]) / -1.0e-12;
        } else {
          v_c[it_circ][j] = (Hw_c[j] - lowest_eigenvalue * w_c[j]) / diff_to_diag;
        }
      }
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) {
        const double diff_to_diag = lowest_eigenvalue - Hoo(j, j);  // diag_elems[j];
        if (std::abs(diff_to_diag) < 1.0e-12) {
          v_o[it_circ][j] = (Hw_o[j] - lowest_eigenvalue * w_o[j]) / -1.0e-12;
        } else {
          v_o[it_circ][j] = (Hw_o[j] - lowest_eigenvalue * w_o[j]) / diff_to_diag;
        }
      }
  
      // Orthogonalize and normalize.
      for (size_t i = 0; i < it_circ; i++) {
        norm = Util::dot_omp(v_c[it_circ], v_c[i]) + Util::dot_omp(v_o[it_circ], v_o[i]);
#pragma omp parallel for
        for (size_t j = 0; j < n_dets; j++) {
          v_c[it_circ][j] -= norm * v_c[i][j];
        }
#pragma omp parallel for
        for (size_t j = 0; j < n_orb_param; j++) {
          v_o[it_circ][j] -= norm * v_o[i][j];
        }
      }
      norm = sqrt(Util::dot_omp(v_c[it_circ], v_c[it_circ]) + Util::dot_omp(v_o[it_circ], v_o[it_circ]));
  
#pragma omp parallel for
      for (size_t j = 0; j < n_dets; j++) {
        v_c[it_circ][j] /= norm;
      }
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) {
        v_o[it_circ][j] /= norm;
      }
      mat_vec(v_c[it_circ], v_o[it_circ], Hv_c[it_circ], Hv_o[it_circ]);
  
      if (norm<1e-12) {
        break;
      }
  
      // Construct subspace matrix.
      for (size_t i = 0; i <= it_circ; i++) {
        h_krylov(i, it_circ) = Util::dot_omp(v_c[i], Hv_c[it_circ]) + Util::dot_omp(v_o[i], Hv_o[it_circ]);
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
      for (size_t j = 0; j < n_dets; j++) {
        double w_j = 0.0;
        double Hw_j = 0.0;
        for (size_t i = 0; i < it_circ + 1; i++) {
          w_j += v_c[i][j] * eigenvector_krylov[i];
          Hw_j += Hv_c[i][j] * eigenvector_krylov[i];
        }
        w_c[j] = w_j;
        Hw_c[j] = Hw_j;
      }
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) {
        double w_j = 0.0;
        double Hw_j = 0.0;
        for (size_t i = 0; i < it_circ + 1; i++) {
          w_j += v_o[i][j] * eigenvector_krylov[i];
          Hw_j += Hv_o[i][j] * eigenvector_krylov[i];
        }
        w_o[j] = w_j;
        Hw_o[j] = Hw_j;
      }
  
      printf("Davidson #%zu: %.10f\n", it_real, lowest_eigenvalue);
      it_real++;
      if (std::abs(lowest_eigenvalue - lowest_eigenvalue_prev) < TOLERANCE && lowest_eigenvalue < 0. && it_real > 10) {
        converged = true;
      } else {
        lowest_eigenvalue_prev = lowest_eigenvalue;
      }
  
      if (converged) break;
    }
mat_vec(w_c, w_o, Hw_c, Hw_o);
for (size_t i=0;i<n_dets; i++) Hw_c[i] -= lowest_eigenvalue * w_c[i];
for (size_t i=0;i<n_orb_param; i++) Hw_o[i] -= lowest_eigenvalue * w_o[i];
std::cout<<"\nresidual norm "<<Util::dot_omp(Hw_c, Hw_c) + Util::dot_omp(Hw_o, Hw_o);
    std::cout<<"\n w_c ";
    for (size_t i =1; i<10; i++) std::cout<<" "<<w_c[i]/w_c[0];
    std::cout<<"\n w_o ";
    for (const auto val: w_o) std::cout<<" "<<val/w_c[0];

    for (size_t j = 0; j < n_orb_param; j++) w_o[j] /= w_c[0];

    Timer::end();
    return w_o;

  }

private:
  size_t n_dets, n_orb_param;

  const double e_var;

  const SparseMatrix& Hamiltonian;

  const Matrix<float, Dynamic, Dynamic, RowMajor>& Hco;
  
  const Matrix<double, Dynamic, Dynamic, RowMajor>& Hoo;
  
  void mat_vec(const std::vector<double>& in_c,
		   const std::vector<double>& in_o,
                   std::vector<double>& out_c,
                   std::vector<double>& out_o) const {

    for (size_t i = 0; i < n_dets; i++) out_c[i] = 0.;
    for (size_t i = 0; i < n_orb_param; i++) out_o[i] = 0.;
    // Hcc
    auto Hcc_xc = Hamiltonian.mul(in_c);
#pragma omp parallel for 
    for (size_t i = 1; i < n_dets; i++) out_c[i] += 2 * (Hcc_xc[i] - e_var * in_c[i]);
    // start from 1 since first row of Hcc is always empty

    // Hco
#pragma omp parallel for 
    for (size_t i = 0; i < n_dets; i++) { // TODO: check col/row-major
      for (size_t j = 0; j < n_orb_param; j++) {
        const double Hco_ij = Hco(i, j);
#pragma omp atomic
        out_c[i] += Hco_ij * in_o[j]; 
#pragma omp atomic
        out_o[j] += Hco_ij * in_c[i];
      }
    }

    // Hoo
#pragma omp parallel for
    for (size_t i = 0; i < n_orb_param; i++) { // TODO: upper triangle or not
      double diff_i = 0.;
      for (size_t j = 0; j < n_orb_param; j++) {
        diff_i += Hoo(i, j) * in_o[j];
      }
#pragma omp atomic
      out_o[i] += diff_i;
    }
  }

};
