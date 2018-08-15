#pragma once

#include <unordered_map>
#include <vector>

#include "k_point.h"

class KPoints {
 public:
  KPoints() {}

  void init(const double r_cut);

  unsigned get_n_orbs() const { return points.size(); }

  std::vector<KPoint> get_k_diffs() const;

  KPoint& operator[](const size_t i) { return points[i]; }

 private:
  std::vector<KPoint> points;

  std::unordered_map<KPoint, size_t, KPointHasher> lut;
};
