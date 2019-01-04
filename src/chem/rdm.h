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
  RDM(Integrals* integrals_ptr) {
    integrals_p = integrals_ptr;
    n_orbs = integrals_ptr->n_orbs;
    n_up = integrals_ptr->n_up;
    n_dn = integrals_ptr->n_dn;
  }

  ~RDM() { integrals_p = nullptr; }

  void get_1rdm(const std::vector<Det>&, const std::vector<double>&, const bool dump_csv = false);

  /*
    size_t nonsym_combine2(const size_t, const size_t) const;  // used for generate_natorb_integrals
    with hash tables

    size_t nonsym_combine4(const size_t, const size_t, const size_t, const size_t) const;
  */

  void get_2rdm_slow(const std::vector<Det>&, const std::vector<double>&);

  void get_2rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const std::vector<std::vector<size_t>>& connections);

  void get_1rdm_from_2rdm();

  void dump_2rdm(const bool dump_csv = false) const;

  double one_rdm_elem(unsigned, unsigned) const;

  double two_rdm_elem(unsigned, unsigned, unsigned, unsigned) const;

 private:
  Integrals* integrals_p;

  size_t n_orbs;

  unsigned n_up, n_dn;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;

  inline size_t combine4_2rdm(size_t p, size_t q, size_t r, size_t s) const;

  int permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const;

  void compute_energy_from_rdm() const;

  void get_2rdm_elements(
      const Det& connected_det,
      const double& connected_coef,
      const Det& this_det,
      const double& this_coef);

  void write_in_2rdm(size_t p, size_t q, size_t r, size_t s, double value);
};
