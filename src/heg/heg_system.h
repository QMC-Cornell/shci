#pragma once

#include "../base_system.h"
#include "k_points.h"

class HegSystem : public BaseSystem {
 public:
  void setup() override;

  void find_connected_dets(
      const Det&,
      const double,
      const double,
      const std::function<void(const Det&, const int)>&) const override;

  double get_hamiltonian_elem(const Det&, const Det&, const int) const override { return 0.0; }

  void update_diag_helper() override {}

 private:
  double r_cut;

  double r_s;

  double k_unit;

  double H_unit;

  KPoints k_points;

  double max_abs_H;

  std::unordered_map<KPoint, std::vector<std::pair<KPoint, double>>, KPointHasher> hci_queue;

  void setup_hci_queue();

  void setup_hf();
};
