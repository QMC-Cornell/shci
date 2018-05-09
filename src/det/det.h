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

  template <class B>
  void serialize(B& buf) const {
    buf << up << dn;
  }

  template <class B>
  void parse(B& buf) {
    buf >> up >> dn;
  }
};
