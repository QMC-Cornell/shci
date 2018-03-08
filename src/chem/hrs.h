#pragma once

class HRS {
 public:
  double H;

  unsigned r;

  unsigned s;

  HRS(const unsigned r, const unsigned s, const double H) : H(H), r(r), s(s) {}
};
