#pragma once

#include <functional>
#include <string>
#include <vector>
#include "det/det.h"

class BaseSystem {
 public:
  int n_up;

  int n_dn;

  int n_elecs;

  std::vector<std::string> dets;

  std::vector<double> coefs;

  double energy_hf;

  double energy_var;

  void setup();

  void setup_variation(){};

  void setup_perturbation(){};

  virtual void find_connected_dets(
      const Det& det,
      const double eps,
      const std::function<void(const Det&)>& connected_det_handler) = 0;

  virtual double get_hamiltonian_elem(const Det& det_i, const Det& det_j) = 0;
};
