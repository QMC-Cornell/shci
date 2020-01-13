#pragma once

#include "integrals.h"
#include "../config.h"
#include <unordered_map>
#include <eigen/Eigen/Dense>

using namespace Eigen;

class FullOptimization {
  // This module is designed such that for each optimization iteration
  // a new Optimization object is instantiated.
public:
  FullOptimization(Integrals *integrals_ptr) {
    integrals_p = integrals_ptr;
    n_orbs = integrals_ptr->n_orbs;
    n_up = integrals_ptr->n_up;
    n_dn = integrals_ptr->n_dn;
    rot = MatrixXd::Zero(n_orbs, n_orbs);
    get_orb_param_indices_in_matrix();
    get_indices2index_hashtable();
    n_dets_truncate = Config::get<size_t>("optimization/n_dets_truncate", 1);
    hessian_ci_orb = MatrixXd::Zero(n_dets_truncate, orb_dim);
  }

  ~FullOptimization() {
    integrals_p = nullptr;
  }

  void get_1rdm(const std::vector<Det>&, const std::vector<double>&, const bool dump_csv = false);

  void get_2rdm(
    const std::vector<Det>&,
    const std::vector<double>&,
    const std::vector<std::vector<size_t>>& connections);

  void generate_optorb_integrals_from_newton(
  	const std::vector<double>& row_sum,
	const std::vector<double>& diag,
	const std::vector<double>& coefs,
	const double E_var);

  void rewrite_integrals();

  MatrixXd get_rotation_matrix() const { return rot; };

private:

  Integrals *integrals_p;

  unsigned n_orbs;

  unsigned n_up, n_dn;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;

  typedef std::vector<std::vector<std::vector<std::vector<double>>>>
      Integrals_array;

  typedef std::pair<unsigned, unsigned> index_t;
  
  // (row, col) indices of orbital parameters in antisymmetric matrix
  std::vector<index_t> orb_param_indices_in_matrix;

  // number of orbital paramters; ie, length of orb_param_indices
  size_t orb_dim;
  
  // number of dets to keep when evaluating CI-orbital block of Hessian
  size_t n_dets_truncate;
  
  MatrixXd hessian_ci_orb;
  
  struct hash_pair { 
    size_t operator()(const index_t& p) const
    { 
        auto hash1 = std::hash<unsigned>{}(p.first); 
        auto hash2 = std::hash<unsigned>{}(p.second); 
        return hash1 ^ hash2; 
    } 
  };
  
  // Hashtable: key=(row,col) indices of orbital parameters in matrix; 
  // value=index in vector orb_param_indices
  std::unordered_map<index_t, size_t, hash_pair> indices2index;
  
  Integrals_array new_integrals;

  MatrixXd rot;
  
  double one_rdm_elem(unsigned, unsigned) const;
  
  double two_rdm_elem(unsigned, unsigned, unsigned, unsigned) const;

  void get_orb_parameter_indices_in_matrix();

  void get_indices2index_hashtable();
  
  double Hessian_ci_orb(const std::vector<Det>& dets, const std::vector<double>& coefs,
    const size_t i_det, const size_t m, const size_t n);
  
  void rotate_integrals();

  void dump_integrals(const Integrals_array &new_integrals,
                      const char *file_name) const;

  //std::vector<index_t> parameter_indices() const;

  void get_orb_param_indices_in_matrix();
  
  inline size_t combine4_2rdm(size_t p, size_t q, size_t r, size_t s) const;
  
  int permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const;

  void get_2rdm_elements(
    const Det& connected_det,
    const double connected_coef,
    const size_t j_det,
    const Det& this_det,
    const double this_coef,
    const size_t i_det);

  void write_in_1rdm_and_hessian(size_t p, size_t q, double perm_fac, size_t i_det, double coef_i, size_t j_det, double coef_j);

  void write_in_2rdm_and_hessian(size_t p, size_t q, size_t r, size_t s, double perm_fac, size_t i_det, double coef_i, size_t j_det, double coef_j);

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
