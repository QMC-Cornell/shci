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

  KPoint operator[](const size_t i) const { return points.at(i); }

  int find(const KPoint& point) const {
    if (lut.count(point) == 1) return lut.at(point);
    return -1;
  }

 private:
  std::vector<KPoint> points;

  std::unordered_map<KPoint, size_t, KPointHasher> lut;
};
