#pragma once

#include <hps/src/hps.h>
#include <unordered_set>

class HalfDet {
 public:
  size_t n_elecs_hf;

  std::unordered_set<size_t> orbs_from;

  std::unordered_set<size_t> orbs_to;

  HalfDet() {}

  HalfDet(const size_t n_elecs_hf) : n_elecs_hf(n_elecs_hf) {}
};

namespace hps {
template <class B>
class Serializer<HalfDet, B> {
 public:
  static void serialize(const HalfDet& det, OutputBuffer<B>& buf) {
    Serializer<size_t, B>::serialize(det.n_elecs_hf, buf);
    Serializer<std::unordered_set<size_t>, B>::serialize(det.orbs_from, buf);
    Serializer<std::unordered_set<size_t>, B>::serialize(det.orbs_to, buf);
  }
  static void parse(HalfDet& det, InputBuffer<B>& buf) {
    Serializer<size_t, B>::parse(det.n_elecs_hf, buf);
    Serializer<std::unordered_set<size_t>, B>::parse(det.orbs_from, buf);
    Serializer<std::unordered_set<size_t>, B>::parse(det.orbs_to, buf);
  }
};
}  // namespace hps
