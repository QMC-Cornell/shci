#pragma once

class RSH {
 public:
  double H;

  unsigned r;

  unsigned s;

  RSH(const unsigned r, const unsigned s, const double H) : r(r), s(s), H(H) {}
};
