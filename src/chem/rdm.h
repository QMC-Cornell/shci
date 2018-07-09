#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "../det/half_det.h"
#include "integrals.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
 public:
  void get_1rdm(const std::vector<Det>&, const std::vector<double>&, const Integrals&);

  void generate_natorb_integrals(const Integrals&) const;

  /*
    size_t nonsym_combine2(const size_t, const size_t) const;  // used for generate_natorb_integrals
    with hash tables

    size_t nonsym_combine4(const size_t, const size_t, const size_t, const size_t) const;
  */

  void get_2rdm(const std::vector<Det>&, const std::vector<double>&, const Integrals&);

 private:
  MatrixXd one_rdm;

  MatrixXd two_rdm;

  unsigned combine4_2rdm(unsigned, unsigned, unsigned, unsigned, unsigned) const;

  int permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const;
};
