#pragma once

#include <unordered_set>

class HalfDet {
 public:
  int n_hf_elecs;
  std::unordered_set<int> v_holes;
  std::unordered_set<int> c_elecs;
};
