#pragma once

#include <cmath>
#include "../util.h"

class UncertResult {
 public:
  double value;

  double uncert;

  UncertResult(const double value = 0.0, const double uncert = 0.0)
      : value(value), uncert(uncert) {}

  UncertResult operator+(const UncertResult& rhs) const {
    UncertResult res;
    res.value = value + rhs.value;
    res.uncert = sqrt(uncert * uncert + rhs.uncert * rhs.uncert);
    return res;
  }

  std::string to_string() const { return Util::str_printf("%.10f +- %.10f", value, uncert); }
};
