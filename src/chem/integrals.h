#pragma once

#include <fgpl/src/hash_map.h>
#include <unordered_map>
#include <vector>
#include "../det/det.h"
#include "hpqrs.h"
#include "integrals_container.h"

class Integrals {
 public:
  double energy_core;

  unsigned n_orbs;

  unsigned n_elecs;

  unsigned n_up;

  unsigned n_dn;

  std::vector<unsigned> orb_sym;

  std::vector<unsigned> orb_order;

  std::vector<unsigned> orb_order_inv;

  Det det_hf;

  void load();

  double get_1b(const unsigned p, const unsigned q) const;

  double get_2b(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  static size_t combine2(const size_t a, const size_t b);

  static size_t combine4(const size_t a, const size_t b, const size_t c, const size_t d);

  template <class B>
  void serialize(B& buf) const;

  template <class B>
  void parse(B& buf);

 private:
  IntegralsContainer integrals_1b;
  IntegralsContainer integrals_2b;

  bool explicit_orbs;

  std::vector<Hpqrs> raw_integrals;

  void read_fcidump();

  std::vector<unsigned> get_adams_syms(const std::vector<int>& orb_syms_raw) const;

  void generate_det_hf();

  std::vector<double> get_orb_energies() const;

  void reorder_orbs(const std::vector<double>& orb_energies);

  bool load_from_cache(const std::string& filename);

  void save_to_cache(const std::string& filename) const;
};

template <class B>
void Integrals::serialize(B& buf) const {
  buf << energy_core << n_orbs << n_elecs << n_up << n_dn << orb_sym << orb_order << orb_order_inv
      << det_hf;
  buf << integrals_1b << integrals_2b;
}

template <class B>
void Integrals::parse(B& buf) {
  buf >> energy_core >> n_orbs >> n_elecs >> n_up >> n_dn >> orb_sym >> orb_order >>
      orb_order_inv >> det_hf;
  buf >> integrals_1b >> integrals_2b;
}
