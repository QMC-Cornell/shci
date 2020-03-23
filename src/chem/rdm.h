#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "../det/half_det.h"
#include "../solver/sparse_matrix.h"
#include "integrals.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
  typedef std::pair<unsigned, unsigned> index_t;
 public:
  RDM(const Integrals& integrals_): integrals(integrals_), 
    n_orbs(integrals_.n_orbs),
    n_up(integrals_.n_up),
    n_dn(integrals_.n_dn),
    time_sym(Config::get<bool>("time_sym", false)) {
  }

  void prepare_for_writing_in_hessian_ci_orb(
      const std::vector<index_t>& parameter_indices,
      MatrixXd* const hessian_ci_orb_p);

  void get_1rdm(const std::vector<Det>&, const std::vector<double>&);
  
  void get_1rdm_unpacked(const std::vector<Det>&, const std::vector<double>&);

  void get_2rdm_slow(const std::vector<Det>&, const std::vector<double>&);

  void get_2rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const std::vector<std::vector<size_t>>& connections);

  void get_2rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const SparseMatrix& hamiltonian_matrix);

  void get_1rdm_from_2rdm();
  
  void dump_1rdm() const;

  void dump_2rdm(const bool dump_csv = false) const;

  double one_rdm_elem(const unsigned, const unsigned) const;

  double two_rdm_elem(const unsigned, const unsigned, const unsigned, const unsigned) const;

  void clear();

 private:
  const Integrals& integrals;

  const unsigned n_orbs, n_up, n_dn;

  const bool time_sym;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;

  MatrixXd* hessian_ci_orb_p = nullptr;

  // Hashtable: key=(row,col) indices of orbital parameters in antisymm matrix; 
  // value=index in vector orb_param_indices
  struct hash_pair { 
    size_t operator()(const index_t& p) const
    { 
        auto hash1 = std::hash<unsigned>{}(p.first); 
        auto hash2 = std::hash<unsigned>{}(p.second); 
        return hash1 ^ hash2; 
    } 
  };
  
  std::unordered_map<index_t, size_t, hash_pair> indices2index;

  inline size_t combine4_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  int permfac_ccaa(HalfDet halfket, const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  void compute_energy_from_rdm() const;

  void get_2rdm_pair(const Det& connected_det, const double connected_coef, 
       const size_t connected_ind, const Det& this_det, const double this_coef, const size_t this_ind);

  void get_2rdm_elements(
      const Det& connected_det,
      const double connected_coef,
      const size_t j_det,
      const Det& this_det,
      const double this_coef,
      const size_t i_det);

  void MPI_Allreduce_2rdm();

  void write_in_1rdm_and_hessian_co(const unsigned p, const unsigned q, const int perm_fac, const size_t i_det, const double coef_i, const size_t j_det, const double coef_j);

  void write_in_2rdm_and_hessian_co(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const int perm_fac, const size_t i_det, const double coef_i, const size_t j_det, const double coef_j);
};
