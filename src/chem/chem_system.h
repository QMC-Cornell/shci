#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"
#include "integrals.h"
#include "point_group.h"
#include "product_table.h"
#include "rsh.h"

class ChemSystem : public BaseSystem {
 public:
  void setup();

  void find_connected_dets(
      const Det& det,
      const double eps,
      const std::function<void(const Det&)>& connected_det_handler);

  double get_hamiltonian_elem(const Det& det_i, const Det& det_j);

 private:
  unsigned n_orbs;

  std::vector<unsigned> orb_syms;

  PointGroup point_group;

  Integrals integrals;

  ProductTable product_table;

  std::vector<std::vector<RSH>> hci_queue;

  void setup_hci_queue();

  PointGroup get_point_group(const std::string& str) const;
};
