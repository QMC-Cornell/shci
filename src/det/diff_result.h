#pragma once

#include <vector>

class DiffResult {
 public:
  std::vector<unsigned> leftOnly;
  std::vector<unsigned> rightOnly;
  int permutation_factor = 1;
};
