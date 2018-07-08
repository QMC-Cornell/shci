#pragma once

#include <hps/src/hps.h>
#include "half_det.h"

class Det {
 public:
  HalfDet up;

  HalfDet dn;

  friend bool operator==(const Det& a, const Det& b) { return a.up == b.up && a.dn == b.dn; }

  friend bool operator!=(const Det& a, const Det& b) { return !(a == b); }

  friend bool operator<(const Det& a, const Det& b) {
    return a.up < b.up || (a.up == b.up && a.dn < b.dn);
  }

  friend bool operator>(const Det& a, const Det& b) {
    return a.up > b.up || (a.up == b.up && a.dn > b.dn);
  }

  void reverse_spin() {
    if (up == dn) return;
    HalfDet tmp = up;
    up = dn;
    dn = tmp;
  }

  template <class B>
  void serialize(B& buf) const {
    buf << up << dn;
  }

  template <class B>
  void parse(B& buf) {
    buf >> up >> dn;
  }
};

class DetHasher {
 public:
  size_t operator()(const Det& det) const {
    size_t seed = det.up.get_hash_value();
    seed ^= det.dn.get_hash_value() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
