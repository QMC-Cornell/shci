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
  void get_1rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const Integrals&,
      const bool dump_csv = false);

  void generate_natorb_integrals(const Integrals&) const;

  /*
    size_t nonsym_combine2(const size_t, const size_t) const;  // used for generate_natorb_integrals
    with hash tables

    size_t nonsym_combine4(const size_t, const size_t, const size_t, const size_t) const;
  */

  void get_2rdm_slow(const std::vector<Det>&, const std::vector<double>&, const Integrals&);

  void get_2rdm(
      const std::vector<Det>&,
      const std::vector<double>&,
      const Integrals&,
      const std::vector<std::vector<size_t>>& connections,
      const bool dump_csv = false);

 private:
  unsigned n_orbs, n_up, n_dn;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;

  unsigned combine4_2rdm(unsigned, unsigned, unsigned, unsigned, unsigned) const;

  int permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const;

  void compute_energy_from_rdm(const Integrals& integrals) const;

  void get_2rdm_elements(
      const Det& connected_det,
      const double& connected_coef,
      const Det& this_det,
      const double& this_coef);

  void write_in_2rdm(unsigned p, unsigned q, unsigned r, unsigned s, double value);
};
