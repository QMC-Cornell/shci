#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"
#include "../solver/sparse_matrix.h"
#include "hrs.h"
#include "integrals.h"
#include "point_group.h"
#include "product_table.h"
#include "sr.h"
#include <eigen/Eigen/Dense>

class ChemSystem : public BaseSystem {
 public:
  void setup(const bool load_integrals_from_file = true) override;

  double find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const int n_excite)>& handler,
      const bool second_rejection = false) const override;

  double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const int n_excite) const override;

  void update_diag_helper() override;

  void post_variation(std::vector<std::vector<size_t>>& connections) override;

  void post_variation_optimization(
      SparseMatrix& hamiltonian_matrix,
      const std::string& method) override;

  void optimization_microiteration(
      SparseMatrix& hamiltonian_matrix,
      const std::string& method) override;

  void variation_cleanup() override;

  void dump_integrals(const char* filename) override;

  double get_e_hf_1b() const override;

 private:
  std::vector<unsigned> orb_sym;

  double max_hci_queue_elem;

  // max singles queue element
  double max_singles_queue_elem;

  std::vector<std::vector<unsigned>> sym_orbs;

  PointGroup point_group;

  Integrals integrals;

  ProductTable product_table;

  std::vector<std::vector<Hrs>> hci_queue;

  // singles queue
  std::vector<std::vector<Sr>> singles_queue;

  Eigen::MatrixXd rotation_matrix;

  // setup sym orbs
  void setup_sym_orbs();

  void setup_hci_queue();

  // setup singles queue
  void setup_singles_queue();

  PointGroup get_point_group(const std::string& str) const;

  void check_group_elements() const;

  double get_hci_queue_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s);

  // get singles queue elements
  double get_singles_queue_elem(const unsigned p, const unsigned r) const;

  double get_hamiltonian_elem_no_time_sym(const Det& det_i, const Det& det_j, int n_excite) const;

  double get_one_body_diag(const Det& det) const;

  double get_two_body_diag(const Det& det) const;

  double get_one_body_single(const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_two_body_single(
      const Det& det_i, const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const;

  double get_s2(std::vector<double>) const;
};
