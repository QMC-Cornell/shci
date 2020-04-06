#pragma once

class CGSolver {

public:
  CGSolver(const SparseMatrix& hamiltonian_matrix,
           const Matrix<float, Dynamic, Dynamic, RowMajor>& hessian_ci_orb, 
           const Matrix<double, Dynamic, Dynamic, RowMajor>& hessian_orb_orb,
           const double energy_var,
           const VectorXd& gradient_orb):
  e_var(energy_var),
  Hamiltonian(hamiltonian_matrix),
  Hco(hessian_ci_orb),
  Hoo(hessian_orb_orb),
  go(gradient_orb) {
    n_dets = hamiltonian_matrix.count_n_rows();
    n_orb_param = gradient_orb.size();

    SelfAdjointEigenSolver<MatrixXd> es(Hoo);
    VectorXd eigenvalues = es.eigenvalues().transpose();
    //std::cout<<"eigenvals\n"<<eigenvalues;
    std::cout<<"\nHoo min_eigenvalue "<<eigenvalues.minCoeff();
    //diag_shift = eigenvalues.minCoeff() > 0. ? 0. : 1e-3;
    //diag_shift = 1e-3;
    //diag_shift = std::max(1e-6,-2*eigenvalues.minCoeff());
    std::cout<<"\nHco.norm() "<<Hco.norm()<<std::endl;
  }

  std::vector<double> solve() const {
    Timer::start("conjugate gradient solver");
    std::vector<double> x_c(n_dets, 0.), x_o(n_orb_param, 0.);
    std::vector<double> r_c(n_dets, 0.), r_o(n_orb_param, 0.);
    std::vector<double> z_c(n_dets, 0.), z_o(n_orb_param, 0.);
    std::vector<double> p_c(n_dets, 0.), p_o(n_orb_param, 0.);
#pragma omp parallel for 
    for (size_t i = 0; i < n_orb_param; i++) {
      x_o[i] = - go(i) / (Hoo(i, i) + diag_shift);
      r_o[i] = -go(i);
    }
    std::vector<double> product_c(n_dets, 0.), product_o(n_orb_param, 0.);
    mat_vec(x_c, x_o, product_c, product_o);
#pragma omp parallel for 
    for (size_t i = 0; i < n_dets; i++) r_c[i] -= product_c[i];
#pragma omp parallel for 
    for (size_t i = 0; i < n_orb_param; i++) r_o[i] -= product_o[i];

#pragma omp parallel for 
    for (size_t i = 0; i < n_dets; i++) z_c[i] = r_c[i] / 2 / (Hamiltonian.get_diag(i) - e_var + diag_shift);
#pragma omp parallel for 
    for (size_t i = 0; i < n_orb_param; i++) z_o[i] = r_o[i] / (Hoo(i, i) + diag_shift);

    p_c = z_c;
    p_o = z_o;

    double r_norm = std::sqrt(Util::dot_omp(r_c, r_c) + Util::dot_omp(r_o, r_o));
    double zr = Util::dot_omp(z_c, r_c) + Util::dot_omp(z_o, r_o);
    
    unsigned i_iter = 0;
    while (r_norm > 1e-6) {
      mat_vec(p_c, p_o, product_c, product_o);
      if (Parallel::is_master()) printf("Iter %d: residual norm: %.8f\n", i_iter, r_norm);
      double alpha = zr / (Util::dot_omp(p_c, product_c) + Util::dot_omp(p_o, product_o));
#pragma omp parallel for 
      for (size_t i = 0; i < n_dets; i++) x_c[i] += alpha * p_c[i];
#pragma omp parallel for 
      for (size_t i = 0; i < n_orb_param; i++) x_o[i] += alpha * p_o[i];
#pragma omp parallel for 
      for (size_t i = 0; i < n_dets; i++) r_c[i] -= alpha * product_c[i];
#pragma omp parallel for 
      for (size_t i = 0; i < n_orb_param; i++) r_o[i] -= alpha * product_o[i];
      r_norm = std::sqrt(Util::dot_omp(r_c, r_c) + Util::dot_omp(r_o, r_o));
#pragma omp parallel for 
      for (size_t i = 0; i < n_dets; i++) z_c[i] = r_c[i] / 2 / (Hamiltonian.get_diag(i) - e_var + diag_shift);
#pragma omp parallel for 
      for (size_t i = 0; i < n_orb_param; i++) z_o[i] = r_o[i] / (Hoo(i, i) + diag_shift);

      double new_zr = Util::dot_omp(z_c, r_c) + Util::dot_omp(z_o, r_o);
      double beta = new_zr / zr;
      zr = new_zr;
#pragma omp parallel for 
      for (size_t i = 0; i < n_dets; i++) p_c[i] = z_c[i] + beta * p_c[i];
#pragma omp parallel for 
      for (size_t i = 0; i < n_orb_param; i++) p_o[i] = z_o[i] + beta * p_o[i];
      i_iter++;
    }
    Timer::end();
    return x_o;
  }

private:
  size_t n_dets, n_orb_param;

  const double e_var;

  const SparseMatrix& Hamiltonian;

  const Matrix<float, Dynamic, Dynamic, RowMajor>& Hco;
  
  const Matrix<double, Dynamic, Dynamic, RowMajor>& Hoo;
  
  const VectorXd& go;

  double diag_shift = 0.;

  void mat_vec(const std::vector<double>& in_c,
		   const std::vector<double>& in_o,
                   std::vector<double>& out_c,
                   std::vector<double>& out_o) const {

    for (size_t i = 0; i < n_dets; i++) out_c[i] = 0.;
    for (size_t i = 0; i < n_orb_param; i++) out_o[i] = 0.;
    // Hcc
    auto Hcc_xc = Hamiltonian.mul(in_c);
#pragma omp parallel for 
    for (size_t i = 0; i < n_dets; i++) out_c[i] += 2 * (Hcc_xc[i] - e_var * in_c[i]);

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
  for (size_t i = 1; i < n_dets; i++) out_c[i] += diag_shift * in_c[i];
  for (size_t i = 0; i < n_orb_param; i++) out_o[i] += diag_shift * in_o[i];
  }

};
