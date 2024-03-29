#pragma once

#include "integrals.h"
#include "rdm.h"
#include <eigen/Eigen/Dense>

using namespace Eigen;

class IntegralsArray {
public:
  void allocate(const unsigned n_orbs_) {
    n_orbs = n_orbs_;
    n_orbs_2 = n_orbs * n_orbs;
    n_orbs_3 = n_orbs_2 * n_orbs;
    array_2b.resize(n_orbs_3 * n_orbs);
    array_1b.resize(n_orbs_2);
  }

  double& get_2b(const unsigned p, const unsigned q, const unsigned r, const unsigned s) {
    const size_t ind = p * n_orbs_3 + q * n_orbs_2 + r * n_orbs + s;
    return array_2b[ind];
  }
  
  double& get_1b(const unsigned p, const unsigned q) {
    const size_t ind = p * n_orbs + q;
    return array_1b[ind];
  }

  const double* data_2b() const { return array_2b.data(); }

  const double* data_1b() const { return array_1b.data(); }

private:
  size_t n_orbs, n_orbs_2, n_orbs_3;

  std::vector<double> array_2b, array_1b;
};

class Optimization {
  // This module is designed such that for each optimization iteration
  // a new Optimization object is instantiated.
public:
  Optimization(Integrals& integrals_, 
		SparseMatrix& hamiltonian_matrix_,
		const std::vector<Det>& dets_,
		const std::vector<std::vector<double>>& wf_coefs_): 
	integrals(integrals_),
	hamiltonian_matrix(hamiltonian_matrix_),
        dets(dets_),
        wf_coefs(wf_coefs_),
        rdm(integrals, dets, wf_coefs),
        n_orbs(integrals_.n_orbs)
  {
    rot = MatrixXd::Zero(n_orbs, n_orbs);
  }

  void get_natorb_rotation_matrix();

  void get_optorb_rotation_matrix_from_newton();

  void get_optorb_rotation_matrix_from_approximate_newton();

  void get_optorb_rotation_matrix_from_grad_descent();

  void get_optorb_rotation_matrix_from_amsgrad();

  void generate_optorb_integrals_from_bfgs();

  void rotate_and_rewrite_integrals();

  MatrixXd rotation_matrix() const { return rot; };

private:
  Integrals& integrals;
  
  SparseMatrix& hamiltonian_matrix;
  
  const std::vector<Det>& dets;

  const std::vector<std::vector<double>>& wf_coefs;

  RDM rdm;

  const unsigned n_orbs;

  typedef std::pair<unsigned, unsigned> index_t;

  typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdR;

  typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfR;

  MatrixXd generalized_Fock_matrix;

  IntegralsArray new_integrals;

  MatrixXd rot;

  void rotate_integrals();

  void rewrite_integrals();

  std::vector<index_t> parameter_indices() const;
  
  void fill_rot_matrix_with_parameters(
      const VectorXd &parameters,
      const std::vector<index_t> &parameter_indices);

  VectorXd gradient(const std::vector<std::pair<unsigned, unsigned>> &);

  void get_generalized_Fock();

  double generalized_Fock_element(const unsigned m, const unsigned n) const;

  MatrixXdR hessian(const std::vector<std::pair<unsigned, unsigned>> &);

  VectorXd
  hessian_diagonal(const std::vector<std::pair<unsigned, unsigned>> &);

  double Y_matrix(const unsigned p, const  unsigned q, const unsigned r, const unsigned s) const;

  double hessian_part(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;
};
