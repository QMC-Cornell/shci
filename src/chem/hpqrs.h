#pragma once

#include <cstdint>

class Hpqrs {
 public:
  double H;

  uint16_t p;

  uint16_t q;

  uint16_t r;

  uint16_t s;

  Hpqrs(const double H, const uint16_t p, const uint16_t q, const uint16_t r, const uint16_t s)
      : H(H), p(p), q(q), r(r), s(s) {}
};
