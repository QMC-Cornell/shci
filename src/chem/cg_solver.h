#pragma once
#include <eigen/Eigen/Dense>

class CGSolver {

public:
  CGSolver(const SparseMatrix& hamiltonian_matrix,
           const MatrixXd& hessian_ci_orb, 
           const MatrixXd& hessian_orb_orb,
           const double E_var,
           const VectorXd& gradient_orb):
  Hamiltonian(hamiltonian_matrix),
  Hco(hessian_ci_orb),
  Hoo(hessian_orb_orb),
  go(gradient_orb) {

  n_ci_param = hamiltonian_matrix.rows.size() - 1;
  n_orb_param = gradient_orb.size();
  // Hcc
  Hcc_vals.resize(n_ci_param+1);

#pragma omp parallel for
    for (size_t i = 0; i < hamiltonian_matrix.rows.size(); i++) {
      Hcc_vals[i].resize(hamiltonian_matrix.rows[i].size());
      Hcc_vals[i][0] = 2. * hamiltonian_matrix.rows[i].get_value(0) - 2. * E_var;
      for (size_t j_id = 1; j_id < hamiltonian_matrix.rows[i].size(); j_id++) {
        double val = hamiltonian_matrix.rows[i].get_value(j_id);
        Hcc_vals[i][j_id] = 2. * val;
      }
    }

    std::cout<<"constructor ends"<<std::endl;
  }

  VectorXd solve() const {
    std::cout<<"solve starts"<<std::endl;
    VectorXd ans = VectorXd::Zero(n_ci_param + n_orb_param);
    VectorXd r = VectorXd::Zero(n_ci_param + n_orb_param);
    for (size_t i = n_ci_param; i < n_ci_param + n_orb_param; i++) {
      ans(i) = - go(i - n_ci_param) / Hoo(i - n_ci_param, i - n_ci_param);
      r(i) = -go(i - n_ci_param);
    }
    r -= mat_vec(ans);

    VectorXd z = VectorXd::Zero(n_ci_param + n_orb_param);
    for (size_t i = 0; i < n_ci_param; i++) z(i) = r(i) / Hcc_vals[i + 1][0];
    for (size_t i = n_ci_param; i < n_ci_param + n_orb_param; i++) z(i) = r(i) / Hoo(i - n_ci_param, i - n_ci_param);

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
  size_t n_ci_param, n_orb_param;

  const SparseMatrix& Hamiltonian;

  const MatrixXd& Hco;
  
  const MatrixXd& Hoo;
  
  const VectorXd& go;

  std::vector<std::vector<float>> Hcc_vals;

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
