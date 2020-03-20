#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "../det/half_det.h"
#include "../solver/sparse_matrix.h"
#include "integrals.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
 public:
  RDM(const Integrals& integrals_): integrals(integrals_) {
    n_orbs = integrals.n_orbs;
    n_up = integrals.n_up;
    n_dn = integrals.n_dn;
  }

  typedef std::pair<unsigned, unsigned> index_t;

  void prepare_for_writing_in_hessian_ci_orb(
      const std::vector<index_t>& parameter_indices,
      MatrixXd* hessian_ci_orb_p);

  void get_1rdm(const std::vector<Det>&, const std::vector<double>&);

  void get_2rdm_slow(const std::vector<Det>&, const std::vector<double>&);

  void get_2rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const std::vector<std::vector<size_t>>& connections);

  void get_1rdm_from_2rdm();
  
  void dump_1rdm() const;

  void dump_2rdm(const bool dump_csv = false) const;

  double one_rdm_elem(unsigned, unsigned) const;

  double two_rdm_elem(unsigned, unsigned, unsigned, unsigned) const;

  void clear();

 private:
  const Integrals& integrals;

  size_t n_orbs;

  unsigned n_up, n_dn;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;

  MatrixXd* hessian_ci_orb_p = nullptr;

//  typedef std::pair<unsigned, unsigned> index_t;
  
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

  inline size_t combine4_2rdm(size_t p, size_t q, size_t r, size_t s) const;

  int permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const;

  void compute_energy_from_rdm() const;

  void get_2rdm_elements(
      const Det& connected_det,
      const double connected_coef,
      const size_t j_det,
      const Det& this_det,
      const double this_coef,
      const size_t i_det);

  void write_in_1rdm_and_hessian_co(size_t p, size_t q, double perm_fac, size_t i_det, double coef_i, size_t j_det, double coef_j);

  void write_in_2rdm_and_hessian_co(size_t p, size_t q, size_t r, size_t s, double perm_fac, size_t i_det, double coef_i, size_t j_det, double coef_j);
};
