#pragma once

#include "integrals.h"
#include "rdm.h"
#include <eigen/Eigen/Dense>

using namespace Eigen;

class Optimization {
  // This module is designed such that for each optimization iteration
  // a new Optimization object is instantiated.
public:
  Optimization(RDM *rdm_ptr, Integrals *integrals_ptr) {
    rdm_p = rdm_ptr;
    integrals_p = integrals_ptr;
    n_orbs = integrals_ptr->n_orbs;
    rot = MatrixXd::Zero(n_orbs, n_orbs);
  }

  ~Optimization() {
    rdm_p = nullptr;
    integrals_p = nullptr;
  }

  void generate_natorb_integrals();

  void generate_optorb_integrals_from_newton();

  void generate_optorb_integrals_from_approximate_newton();

  void generate_optorb_integrals_from_grad_descent();

  void generate_optorb_integrals_from_amsgrad();

  void dump_integrals(const char *file_name) const;

  void rewrite_integrals();

  MatrixXd get_rotation_matrix() const { return rot; };

private:
  RDM *rdm_p;

  Integrals *integrals_p;

  unsigned n_orbs;

  typedef std::vector<std::vector<std::vector<std::vector<double>>>>
      Integrals_array;

  typedef std::pair<unsigned, unsigned> index_t;

  Integrals_array new_integrals;

  MatrixXd rot;

  void rotate_integrals();

  std::vector<index_t> parameter_indices() const;

  VectorXd find_overshooting_stepsize(double dim,
                                      const VectorXd &new_param) const;

  void fill_rot_matrix_with_parameters(
      const VectorXd &parameters,
      const std::vector<index_t> &parameter_indices);

  VectorXd gradient(const std::vector<std::pair<unsigned, unsigned>> &) const;

  double generalized_Fock(unsigned m, unsigned n) const;

  MatrixXd hessian(const std::vector<std::pair<unsigned, unsigned>> &) const;

  VectorXd
  hessian_diagonal(const std::vector<std::pair<unsigned, unsigned>> &) const;

  double Y_matrix(unsigned p, unsigned q, unsigned r, unsigned s) const;

  double hessian_part(unsigned p, unsigned q, unsigned r, unsigned s) const;
};
