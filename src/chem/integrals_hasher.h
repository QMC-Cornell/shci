#pragma once
#include <cstddef>

class IntegralsHasher {
 public:
  size_t operator()(const size_t key) const { return (key << 19) - key; }
};
