#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"
#include "hrs.h"
#include "integrals.h"
#include "point_group.h"
#include "product_table.h"

class ChemSystem : public BaseSystem {
 public:
  void setup();

  void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const double)>& connected_det_handler) const;

  double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const unsigned excitation_level) const;

 private:
  unsigned n_orbs;

  bool time_sym;

  int z;  // reflection (parity).

  std::vector<unsigned> orb_sym;

  double max_hci_queue_elem;

  //  std::vector<unsigned> sym_n_orbs;

  std::vector<std::vector<unsigned>> sym_orbs;

  PointGroup point_group;

  Integrals integrals;

  ProductTable product_table;

  std::vector<std::vector<Hrs>> hci_queue;

  void setup_hci_queue();

  PointGroup get_point_group(const std::string& str) const;

  double get_hci_queue_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s);

  double get_two_body_double(const Det& det_i, const Det& det_j, const bool no_sign = false);
};
