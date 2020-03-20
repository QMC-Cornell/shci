#pragma once

#include <fgpl/src/hash_map.h>
#include <hps/src/hps.h>
#include <functional>
#include <string>
#include <vector>
#include "det/det.h"
#include "solver/sparse_matrix.h"
#include "system_type.h"
#include "util.h"

class BaseSystem {
 public:
  SystemType type;

  unsigned n_up = 0;

  unsigned n_dn = 0;

  unsigned n_orbs = 0;

  unsigned n_elecs = 0;

  bool time_sym = false;

  bool has_single_excitation = true;

  bool has_double_excitation = true;

  double energy_hf = 0.0;

  double energy_var = 0.0;

  size_t helper_size = 0;

  std::vector<Det> dets;

  std::vector<double> coefs;

  fgpl::HashMap<HalfDet, double, HalfDetHasher> diag_helper;

  size_t get_n_dets() const { return dets.size(); }

  virtual void setup(const bool){};

  virtual void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const int n_excite)>& handler) const = 0;

  virtual double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const int n_excite) const = 0;

  virtual void update_diag_helper() = 0;

  virtual void post_variation(std::vector<std::vector<size_t>>&){};

  virtual void post_variation_optimization(
      SparseMatrix&, const std::string&) {};

  virtual void dump_integrals(const char*){};

  virtual void post_perturbation(){};

  double get_hamiltonian_elem_time_sym(
      const Det& det_i, const Det& det_j, const int n_excite) const {
    double h = get_hamiltonian_elem(det_i, det_j, n_excite);
    if (det_i.up == det_i.dn) {
      if (det_j.up != det_j.dn) h *= Util::SQRT2;
    } else {
      if (det_j.up == det_j.dn) {
        h *= Util::SQRT2;
      } else {
        Det det_i_rev = det_i;
        det_i_rev.reverse_spin();
        h += get_hamiltonian_elem(det_i_rev, det_j, -1);
      }
    }
    return h;
  }

  void unpack_time_sym() {
    const size_t n_dets_old = get_n_dets();
    for (size_t i = 0; i < n_dets_old; i++) {
      const auto& det = dets[i];
      if (det.up < det.dn) {
        Det det_rev = det;
        det_rev.reverse_spin();
        const double coef_new = coefs[i] * Util::SQRT2_INV;
        coefs[i] = coef_new;
        dets.push_back(det_rev);
        coefs.push_back(coef_new);
      }
    }
  }

  template <class B>
  void serialize(B& buf) const {
    buf << n_up << n_dn << dets << coefs << energy_hf << energy_var;
  }

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> dets >> coefs >> energy_hf >> energy_var;
  }

  virtual void variation_cleanup(){};
};
