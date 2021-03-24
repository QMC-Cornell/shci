#pragma once

class LanczosSolver {

public:
  LanczosSolver(const SparseMatrix& hamiltonian_matrix,
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
    Timer::start("Lanczos diagonalization solver");

    const double TOLERANCE = 1e-8;
    const size_t n_iterations_store = 500;
    double lowest_eigenval_prev = 1e5;

    double beta = 0.;
    double alpha = 0.;
    std::vector<double> q_c(wf_coefs), q_o(n_orb_param, 1.);
    std::vector<double> q_c_prev(n_dets, 0.), q_o_prev(n_orb_param, 0.);
    double norm = sqrt(Util::dot_omp(q_c, q_c) + Util::dot_omp(q_o, q_o));
#pragma omp parallel for
    for (size_t j = 0; j < n_dets; j++) q_c[j] = q_c[j] / norm;
#pragma omp parallel for
    for (size_t j = 0; j < n_orb_param; j++) q_o[j] = q_o[j] / norm;
   
    MatrixXd T(n_iterations_store, n_iterations_store); 
    for (unsigned i=0; i<n_iterations_store-1; i++) {
      std::vector<double> v_c(n_dets), v_o(n_orb_param);
      mat_vec(q_c, q_o, v_c, v_o);
//std::cout<<"\n\tv_c "<<v_c[0]<<" "<<v_c[1]<<" v_o "<<v_o[0]<<" "<<v_o[1];
#pragma omp parallel for
      for (size_t j = 0; j < n_dets; j++) v_c[j] = v_c[j] - beta * q_c_prev[j];
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) v_o[j] = v_o[j] - beta * q_o_prev[j];
      alpha = Util::dot_omp(q_c, v_c) + Util::dot_omp(q_o, v_o);
//std::cout<<"\n\talpha "<<alpha;
#pragma omp parallel for
      for (size_t j = 0; j < n_dets; j++) v_c[j] = v_c[j] - alpha * q_c[j];
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) v_o[j] = v_o[j] - alpha * q_o[j];
      beta = sqrt(Util::dot_omp(v_c, v_c) + Util::dot_omp(v_o, v_o));
//      std::cout<<"\n\tbeta "<<beta;
      if (beta < 1e-9) break;
      q_c_prev = q_c; q_o_prev = q_o;
#pragma omp parallel for
      for (size_t j = 0; j < n_dets; j++) q_c[j] = v_c[j] / beta;
#pragma omp parallel for
      for (size_t j = 0; j < n_orb_param; j++) q_o[j] = v_o[j] / beta;
      T(i, i) = alpha;
      T(i, i+1) = beta;
      T(i+1, i) = beta;
 
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(T.leftCols(i+1).topRows(i+1));
      double lowest_eigenval = eigenSolver.eigenvalues().minCoeff();
      std::cout<<"\nIter "<<i<<": "<<lowest_eigenval;
      if (lowest_eigenval_prev - lowest_eigenval < TOLERANCE) break;
      lowest_eigenval_prev = lowest_eigenval;
    }
    return q_o;
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
