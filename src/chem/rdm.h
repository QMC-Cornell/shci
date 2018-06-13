#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "chem_system.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
 public:
  void get_1rdm(const std::vector<Det>&, const std::vector<double>&, const Integrals&);

  void generate_natorb_integrals(const Integrals&) const;

 private:
  MatrixXd one_rdm;
};
