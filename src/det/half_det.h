#pragma once

#include <hps/src/hps.h>
#include <set>

class HalfDet {
 public:
  HalfDet() {}

  HalfDet(const unsigned n_elecs_hf) : n_elecs_hf(n_elecs_hf) {}

  std::vector<unsigned> get_occupied_orbs() const;

  void set(const unsigned orb);

  void unset(const unsigned orb);

  std::vector<unsigned> get_diff_orbs(const HalfDet& det) const;

  template <class B>
  void serialize(hps::OutputBuffer<B>& buf) const;

  template <class B>
  void parse(hps::InputBuffer<B>& buf);

 private:
  unsigned n_elecs_hf;

  std::set<unsigned> orbs_from;

  std::set<unsigned> orbs_to;

  std::vector<unsigned> get_set_diff(
      const std::set<unsigned>& a, const std::set<unsigned>& b) const;
};

namespace hps {
template <class B>
class Serializer<HalfDet, B> {
 public:
  static void serialize(const HalfDet& det, OutputBuffer<B>& buf) { det.serialize(buf); }
  static void parse(HalfDet& det, InputBuffer<B>& buf) { det.parse(buf); }
};
}  // namespace hps
