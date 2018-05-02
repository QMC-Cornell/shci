#pragma once

#include "../base_system.h"

class HegSystem : public BaseSystem {
 public:
  void setup() {}

  void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const double)>& connected_det_handler) const {}

  double get_hamiltonian_elem(const Det& det_i, const Det& det_j, const int n_excite = -1) const {
    return 0.0;
  }
};
