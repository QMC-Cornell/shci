#include <cstddef>

class SquareHasher {
 public:
  size_t operator()(const size_t key) const { return key * key; }
};
