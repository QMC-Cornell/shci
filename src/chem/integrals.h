#pragma once

#include <fgpl/src/hash_map.h>
#include <unordered_map>
#include <vector>
#include "../det/det.h"

class Integrals {
 public:
  double energy_core;

  unsigned n_orbs;

  unsigned n_elecs;

  unsigned n_up;

  unsigned n_dn;

  std::vector<unsigned> orb_sym;

  Det det_hf;

  void load();

  double get_1b(const unsigned p, const unsigned q) const;

  double get_2b(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  static size_t combine2(const size_t a, const size_t b);

  static size_t combine4(const size_t a, const size_t b, const size_t c, const size_t d);

 private:
  fgpl::HashMap<size_t, double> integrals_1b;

  fgpl::HashMap<size_t, double> integrals_2b;

  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, double>> raw_integrals;

  void read_fcidump();

  std::vector<unsigned> get_adams_syms(const std::vector<int>& orb_syms_raw) const;

  void generate_det_hf();

  std::vector<double> get_orb_energies() const;

  void reorder_orbs(const std::vector<double>& orb_energies);
};
