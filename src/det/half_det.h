#pragma once

#include <hps/src/hps.h>
#include <set>

class HalfDet {
 public:
  HalfDet(const unsigned n_elecs_hf = 0) : n_elecs_hf(n_elecs_hf) {}

  std::vector<unsigned> get_occupied_orbs() const;

  void set(const unsigned orb);

  void unset(const unsigned orb);

  bool has(const unsigned orb) const;

  // std::vector<unsigned> get_diff_orbs(const HalfDet& det) const;

  std::pair<std::vector<unsigned>, std::vector<unsigned>> diff(const HalfDet& det) const;

  template <class B>
  void serialize(hps::OutputBuffer<B>& buf) const;

  template <class B>
  void parse(hps::InputBuffer<B>& buf);

 private:
  unsigned n_elecs_hf;

  std::set<unsigned> orbs_from;

  std::set<unsigned> orbs_to;

  // std::vector<unsigned> get_set_diff(
  //     const std::set<unsigned>& a, const std::set<unsigned>& b) const;

  std::pair<std::vector<unsigned>, std::vector<unsigned>> diff_set(
      const std::set<unsigned>& a, const std::set<unsigned>& b) const;
};

template <class B>
void HalfDet::serialize(hps::OutputBuffer<B>& buf) const {
  hps::Serializer<unsigned, B>::serialize(n_elecs_hf, buf);
  hps::Serializer<std::set<unsigned>, B>::serialize(orbs_from, buf);
  hps::Serializer<std::set<unsigned>, B>::serialize(orbs_to, buf);
}

template <class B>
void HalfDet::parse(hps::InputBuffer<B>& buf) {
  hps::Serializer<unsigned, B>::parse(n_elecs_hf, buf);
  hps::Serializer<std::set<unsigned>, B>::parse(orbs_from, buf);
  hps::Serializer<std::set<unsigned>, B>::parse(orbs_to, buf);
}

namespace hps {
template <class B>
class Serializer<HalfDet, B> {
 public:
  static void serialize(const HalfDet& det, OutputBuffer<B>& buf) { det.serialize(buf); }
  static void parse(HalfDet& det, InputBuffer<B>& buf) { det.parse(buf); }
};
}  // namespace hps
