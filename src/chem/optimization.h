#pragma once

#include "integrals.h"
#include "rdm.h"
#include <eigen/Eigen/Dense>

using namespace Eigen;

class Optimization {
  // This module is designed such that for each optimization iteration
  // a new Optimization object is instantiated.
public:
  Optimization(Integrals& integrals_, 
		SparseMatrix& hamiltonian_matrix_,
		const std::vector<Det>& dets_,
		const std::vector<double>& wf_coefs_): 
	integrals(integrals_),
	hamiltonian_matrix(hamiltonian_matrix_),
        rdm(integrals),
        dets(dets_),
        wf_coefs(wf_coefs_),
        n_orbs(integrals_.n_orbs)
  {
    rot = MatrixXd::Zero(n_orbs, n_orbs);
  }

  void get_natorb_rotation_matrix();

  void get_optorb_rotation_matrix_from_newton();

  void get_optorb_rotation_matrix_from_approximate_newton();

  void get_optorb_rotation_matrix_from_grad_descent();

  void get_optorb_rotation_matrix_from_amsgrad();

  void get_optorb_rotation_matrix_from_full_optimization(const double e_var);

  void rotate_and_rewrite_integrals();

  MatrixXd rotation_matrix() const { return rot; };

private:
  Integrals& integrals;
  
  SparseMatrix& hamiltonian_matrix;

  RDM rdm;

  const std::vector<Det>& dets;

  const std::vector<double>& wf_coefs;

  const unsigned n_orbs;

  typedef std::vector<std::vector<std::vector<std::vector<double>>>>
      Integrals_array;

  typedef std::pair<unsigned, unsigned> index_t;

  typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdR;

  typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfR;

  MatrixXd generalized_Fock_matrix;

  Integrals_array new_integrals;

  MatrixXd rot;

  void rotate_integrals();

  void rewrite_integrals();

  std::vector<index_t> parameter_indices() const;
  
  std::vector<index_t> get_most_important_parameter_indices(
        const VectorXd& gradient,
        const MatrixXdR& hessian,
	const std::vector<index_t>& parameter_indices,
        const double parameter_proportion) const;

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
