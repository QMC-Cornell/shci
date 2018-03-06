#pragma once

#include <unordered_map>
#include <vector>

class Integrals {
 public:
  void load_fcidump();

  double energy_core;

  unsigned n_orbs;

  unsigned n_elecs;

  std::vector<int> orb_syms;

  int isym;

  double get_integral_1b(const unsigned p, const unsigned q) const;

  double get_integral_2b(
      const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

 private:
  std::unordered_map<size_t, double> integrals_1b;

  std::unordered_map<size_t, double> integrals_2b;
};
