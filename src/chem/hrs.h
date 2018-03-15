#pragma once

class HRS {
 public:
  double H;

  unsigned r;

  unsigned s;

  HRS(const double H, const unsigned r, const unsigned s) : H(H), r(r), s(s) {}
};
