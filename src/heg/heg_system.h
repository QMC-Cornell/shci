#pragma once

#include "../base_system.h"

class HegSystem : public BaseSystem {
 public:
  void setup() {}

  void find_connected_dets(
      const Det&,
      const double,
      const double,
      const std::function<void(const Det&, const double)>&) const {}

  double get_hamiltonian_elem(const Det&, const Det&, const int n_excite = -1) const {
    return n_excite;
  }
};
