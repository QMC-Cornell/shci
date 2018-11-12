#include "k_points.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

void KPoints::init(const double r_cut) {
  const int8_t n_max = static_cast<int8_t>(std::floor(r_cut));
  const double r_cut_square = r_cut * r_cut;
  for (int i = -n_max; i <= n_max; i++) {
    for (int j = -n_max; j <= n_max; j++) {
      for (int k = -n_max; k <= n_max; k++) {
        if (i * i + j * j + k * k > r_cut_square) continue;
        points.push_back(KPoint(i, j, k));
      }
    }
  }

  std::stable_sort(points.begin(), points.end(), [](const KPoint& a, const KPoint& b) -> bool {
    return a.squared_norm() < b.squared_norm();
  });

  for (size_t i = 0; i < points.size(); i++) {
    lut[points[i]] = i;
  }
}

std::vector<KPoint> KPoints::get_k_diffs() const {
  // Generate all possible differences between two different k points.
  std::unordered_set<KPoint, KPointHasher> k_diff_set;
  std::vector<KPoint> k_diffs;
  const int n = get_n_orbs();
  for (int p = 0; p < n; p++) {
    for (int q = 0; q < n; q++) {
      if (p == q) continue;
      const auto& diff_pq = points[q] - points[p];
      if (k_diff_set.count(diff_pq) == 1) continue;
      k_diffs.push_back(diff_pq);
      k_diff_set.insert(diff_pq);
    }
  }

  // Sort k_diffs into ascending order so that later sorting hci queue will be faster.
  std::stable_sort(k_diffs.begin(), k_diffs.end(), [](const KPoint& a, const KPoint& b) -> bool {
    return a.squared_norm() < b.squared_norm();
  });

  return k_diffs;
}
