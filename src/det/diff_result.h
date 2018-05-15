#pragma once

#include <array>

class DiffResult {
 public:
  std::array<unsigned, 2> left_only;
  std::array<unsigned, 2> right_only;
  unsigned n_diffs = 0;
  int permutation_factor = 1;
};
