#pragma once

#include "../base_system.h"

class HegSystem : public BaseSystem {
 public:
  void setup() override {}

  void find_connected_dets(
      const Det&,
      const double,
      const double,
      const std::function<void(const Det&, const int)>&) const override {}

  double get_hamiltonian_elem(const Det&, const Det&, const int) const override { return 0.0; }

  double get_hamiltonian_elem_no_time_sym(const Det&, const Det&, const int) const override { return 0.0; }

  void update_diag_helper() override {}
};
