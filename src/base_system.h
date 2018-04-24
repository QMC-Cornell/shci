#pragma once

#include <hps/src/hps.h>
#include <functional>
#include <string>
#include <vector>
#include "det/det.h"

class BaseSystem {
 public:
  unsigned n_up;

  unsigned n_dn;

  unsigned n_elecs;

  std::vector<std::string> dets;

  std::vector<double> coefs;

  double energy_hf;

  double energy_var;

  size_t get_n_dets() const { return dets.size(); }

  virtual void setup() = 0;

  virtual void find_connected_dets(
      const Det& det,
      const double eps_max,
      const double eps_min,
      const std::function<void(const Det&, const double)>& connected_det_handler) const = 0;

  virtual double get_hamiltonian_elem(
      const Det& det_i, const Det& det_j, const unsigned excitation_level) const = 0;

  template <class B>
  void serialize(hps::OutputBuffer<B>& buf) const {
    hps::Serializer<unsigned, B>::serialize(n_up, buf);
    hps::Serializer<unsigned, B>::serialize(n_dn, buf);
    hps::Serializer<std::vector<std::string>, B>::serialize(dets, buf);
    hps::Serializer<std::vector<double>, B>::serialize(coefs, buf);
    hps::Serializer<double, B>::serialize(energy_hf, buf);
    hps::Serializer<double, B>::serialize(energy_var, buf);
  }

  template <class B>
  void parse(hps::InputBuffer<B>& buf) {
    hps::Serializer<unsigned, B>::parse(n_up, buf);
    hps::Serializer<unsigned, B>::parse(n_dn, buf);
    n_elecs = n_up + n_dn;
    hps::Serializer<std::vector<std::string>, B>::parse(dets, buf);
    hps::Serializer<std::vector<double>, B>::parse(coefs, buf);
    hps::Serializer<double, B>::parse(energy_hf, buf);
    hps::Serializer<double, B>::parse(energy_var, buf);
  }
};

namespace hps {
template <class S, class B>
class Serializer<S, B, typename std::enable_if<std::is_base_of<BaseSystem, S>::value, void>::type> {
 public:
  static void serialize(const S& system, hps::OutputBuffer<B>& buf) { system.serialize(buf); }
  static void parse(S& system, hps::InputBuffer<B>& buf) { system.parse(buf); }
};
}  // namespace hps
