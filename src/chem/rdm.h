#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "integrals.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
 public:
  void get_1rdm(const std::vector<Det>&, const std::vector<double>&, const Integrals&);

  void generate_natorb_integrals(const Integrals&) const;

  /*
  size_t nonsym_combine2(
      const size_t a, const size_t b);  // used for generate_natorb_integrals with hash tables

  size_t nonsym_combine4(const size_t a, const size_t b, const size_t c, const size_t d);
  */

 private:
  MatrixXd one_rdm;
};
