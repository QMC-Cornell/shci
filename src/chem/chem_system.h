#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"
#include "../solver/sparse_matrix.h"
#include "hrs.h"
#include "integrals.h"
#include "point_group.h"
#include "product_table.h"

class ChemSystem : public BaseSystem {
 public:
  void setup() override;

  void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const int n_excite)>& handler) const override;

  double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const int n_excite) const override;

  void update_diag_helper() override;

  void post_variation(const std::vector<std::vector<size_t>>& connections) override;

 private:
  std::vector<unsigned> orb_sym;

  double max_hci_queue_elem;

  std::vector<std::vector<unsigned>> sym_orbs;

  PointGroup point_group;

  Integrals integrals;

  ProductTable product_table;

  std::vector<std::vector<Hrs>> hci_queue;

  void setup_hci_queue();

  PointGroup get_point_group(const std::string& str) const;

  double get_hci_queue_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s);

  double get_hamiltonian_elem_no_time_sym(const Det& det_i, const Det& det_j, int n_excite) const;

  double get_one_body_diag(const Det& det) const;

  double get_two_body_diag(const Det& det) const;

  double get_one_body_single(const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_two_body_single(
      const Det& det_i, const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_s2() const;
};
