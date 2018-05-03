#pragma once

#include <hps/src/hps.h>
#include <functional>
#include <string>
#include <vector>
#include "det/det.h"

class BaseSystem {
 public:
  unsigned n_up;

  unsigned n_dn;

  unsigned n_elecs;

  double energy_hf;

  double energy_var;

  std::vector<std::string> det_strs;

  std::vector<double> coefs;

  Det get_det(const size_t i) const { return hps::from_string<Det>(det_strs[i]); }

  size_t get_n_dets() const { return det_strs.size(); }

  virtual void setup() = 0;

  virtual void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const double)>& connected_det_handler) const = 0;

  virtual double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const int n_excite = -1) const = 0;

  template <class B>
  void serialize(B& buf) const {
    buf << n_up << n_dn << det_strs << coefs << energy_hf << energy_var;
  }

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> det_strs >> coefs >> energy_hf >> energy_var;
  }
};
