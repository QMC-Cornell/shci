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
        wf_coefs(wf_coefs_)
  {
    n_orbs = integrals.n_orbs;
    rot = MatrixXd::Zero(n_orbs, n_orbs);
  }

  void generate_natorb_integrals();

  void generate_optorb_integrals_from_newton();

  void generate_optorb_integrals_from_approximate_newton();

  void generate_optorb_integrals_from_grad_descent();

  void generate_optorb_integrals_from_amsgrad();

  void generate_optorb_integrals_from_full_optimization(const double e_var);

  void dump_integrals(const char *file_name) const;

  void rewrite_integrals();

  MatrixXd get_rotation_matrix() const { return rot; };

private:
  Integrals& integrals;
  
  SparseMatrix& hamiltonian_matrix;

  RDM rdm;

  const std::vector<Det>& dets;

  const std::vector<double>& wf_coefs;

  unsigned n_orbs;

  typedef std::vector<std::vector<std::vector<std::vector<double>>>>
      Integrals_array;

  typedef std::pair<unsigned, unsigned> index_t;

  MatrixXd generalized_Fock_matrix;

  Integrals_array new_integrals;

  MatrixXd rot;

  void rotate_integrals();

  void dump_integrals(const Integrals_array &new_integrals,
                      const char *file_name) const;

  std::vector<index_t> parameter_indices() const;
  
  std::vector<index_t> get_most_important_parameter_indices(
        const VectorXd& gradient,
        const MatrixXd& hessian,
	const std::vector<index_t>& parameter_indices,
        const double parameter_proportion) const;

  VectorXd find_overshooting_stepsize(double dim,
                                      const VectorXd &new_param) const;

  void fill_rot_matrix_with_parameters(
      const VectorXd &parameters,
      const std::vector<index_t> &parameter_indices);

  VectorXd gradient(const std::vector<std::pair<unsigned, unsigned>> &);

  void get_generalized_Fock();

  double generalized_Fock_element(unsigned m, unsigned n) const;

  MatrixXd hessian(const std::vector<std::pair<unsigned, unsigned>> &);

  VectorXd
  hessian_diagonal(const std::vector<std::pair<unsigned, unsigned>> &);

  double Y_matrix(unsigned p, unsigned q, unsigned r, unsigned s) const;

  double hessian_part(unsigned p, unsigned q, unsigned r, unsigned s) const;
};
