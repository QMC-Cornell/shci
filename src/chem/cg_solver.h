#pragma once
#include <eigen/Eigen/Dense>

class CGSolver {

public:
  CGSolver(const SparseMatrix& hamiltonian_matrix,
           const MatrixXd& hessian_ci_orb, 
           const MatrixXd& hessian_orb_orb,
           const double energy_var,
           const VectorXd& gradient_orb):
  Hamiltonian(hamiltonian_matrix),
  Hco(hessian_ci_orb),
  Hoo(hessian_orb_orb),
  e_var(energy_var)
  go(gradient_orb) {

    n_dets = hamiltonian_matrix.get_n_rows();
    n_orb_param = gradient_orb.size();
  }

  VectorXd solve() const {
    std::cout<<"solve starts"<<std::endl;
    std::vector<double> res_c(n_dets);
    std::vector<double> res_o(n_orb_param, 0.);
    VectorXd r = VectorXd::Zero(n_dets + n_orb_param);
    for (size_t i = 0; i < n_orb_param; i++) {
      res_c(i) = - go(i) / Hoo(i, i);
      r(n_dets + i) = -go(i);
    }
    r -= mat_vec(res_c, res_o);

    VectorXd z = VectorXd::Zero(n_dets + n_orb_param);
    for (size_t i = 1; i < n_dets; i++) z(i) = r(i) / 2 / (Hamiltonian.get_diag(i) - e_var);
    for (size_t i = 0; i < n_orb_param; i++) z(n_dets + i) = r(n_dets + i) / Hoo(i, i);

    VectorXd p = z;
std::cout<<"solver initialized"<<std::endl;

    while (r.norm() > 1e-6) {
      VectorXd Ap = mat_vec(p);
      double old_zr = z.dot(r);
      std::cout<<"r.norm "<<r.norm()<<std::endl;
      double alpha = old_zr / (p.dot(Ap));
      ans += alpha * p;
      r -= alpha * Ap;

      for (size_t i = 0; i < n_ci_param; i++) z(i) = r(i) / Hcc_vals[i + 1][0];
      for (size_t i = n_ci_param; i < n_ci_param + n_orb_param; i++) z(i) = r(i) / Hoo(i - n_ci_param, i - n_ci_param);

      double beta = z.dot(r) / old_zr;
      p *= beta;
      p += z;
    }
//std::cout<<"\n"<<ans; std::exit(0);
    return ans.tail(n_orb_param);
  }

private:
  size_t n_dets, n_orb_param;

  double e_var;

  const SparseMatrix& Hamiltonian;

  const MatrixXd& Hco;
  
  const MatrixXd& Hoo;
  
  const VectorXd& go;

  VectorXd mat_vec(const VectorXd& p) const {
    VectorXd ans = VectorXd::Zero(n_ci_param + n_orb_param);
    // Hcc
#pragma omp parallel for
    for (size_t i = 1; i <= n_ci_param; i++) {
      double diff_i = Hcc_vals[i][0] * p(i - 1);
      for (size_t j_id = 1; j_id < Hamiltonian.rows[i].size(); j_id++) {
        const size_t j = Hamiltonian.rows[i].get_index(j_id);
        const double Hcc_ij = Hcc_vals[i][j_id];
        diff_i += Hcc_ij * p(j - 1);
        double diff_j = Hcc_ij * p(i - 1);
#pragma omp atomic
        ans(j - 1) += diff_j;
      }
#pragma omp atomic
      ans(i - 1) += diff_i;
    }

    // Hco
#pragma omp parallel for
    for (size_t i = 1; i <= n_ci_param; i++) {
      for (size_t j = 0; j < n_orb_param; j++) {
        const double Hco_ij = Hco(i, j);
#pragma omp atomic
        ans(i - 1) += Hco_ij * p(n_ci_param + j); 
#pragma omp atomic
        ans(n_ci_param + j) += Hco_ij * p(i - 1);
      }
    }

    // Hoo
#pragma omp parallel for
    for (size_t i = 0; i < n_orb_param; i++) {
      double diff_i = 0;
      for (size_t j = 0; j < n_orb_param; j++) {
        diff_i += Hoo(i, j) * p(n_ci_param + j);
      }
#pragma omp atomic
      ans(n_ci_param + i) += diff_i;
    }
    
    return ans;
  }

};
