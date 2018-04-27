#pragma once

#include <hps/src/hps.h>
#include "half_det.h"

class Det {
 public:
  HalfDet up;

  HalfDet dn;

  Det(const unsigned n_up_hf = 0, const unsigned n_dn_hf = 0)
      : up(HalfDet(n_up_hf)), dn(HalfDet(n_dn_hf)) {}

  friend bool operator==(const Det& a, const Det& b) {
    return a.up == b.up && a.dn == b.dn;
  }

  friend bool operator!=(const Det& a, const Det& b) { return !(a == b); }

  friend bool operator<(const Det& a, const Det& b) {
    return a.up < b.up || (a.up == b.up && a.dn < b.dn);
  }

  friend bool operator>(const Det& a, const Det& b) {
    return a.up > b.up || (a.up == b.up && a.dn > b.dn);
  }
};

namespace hps {
template <class B>
class Serializer<Det, B> {
 public:
  static void serialize(const Det& det, OutputBuffer<B>& buf) {
    Serializer<HalfDet, B>::serialize(det.up, buf);
    Serializer<HalfDet, B>::serialize(det.dn, buf);
  }
  static void parse(Det& det, InputBuffer<B>& buf) {
    Serializer<HalfDet, B>::parse(det.up, buf);
    Serializer<HalfDet, B>::parse(det.dn, buf);
  }
};
}  // namespace hps
