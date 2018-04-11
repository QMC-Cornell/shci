#pragma once

#include <functional>
#include <string>
#include <vector>
#include "det/det.h"

class BaseSystem {
 public:
  unsigned n_up;

  unsigned n_dn;

  unsigned n_elecs;

  std::vector<std::string> dets;

  std::vector<double> coefs;

  double energy_hf;

  double energy_var;

  virtual void setup() = 0;

  virtual void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const double)>& connected_det_handler) const = 0;

  virtual double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const unsigned excitation_level) const = 0;
};
