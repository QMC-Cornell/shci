#pragma once

#include <array>
#include <cstdint>

class KPoint {
 public:
  int8_t x;

  int8_t y;

  int8_t z;

  KPoint(){};

  KPoint(int8_t x, int8_t y, int8_t z) : x(x), y(y), z(z) {}

  size_t squared_norm() const;

 private:
  friend KPoint operator-(const KPoint& lhs, const KPoint& rhs);

  friend KPoint operator+(const KPoint& lhs, const KPoint& rhs);

  friend bool operator==(const KPoint& lhs, const KPoint& rhs);

  friend bool operator!=(const KPoint& lhs, const KPoint& rhs);

  friend bool operator==(const KPoint& lhs, const int rhs);
};

class KPointHasher {
 public:
  size_t operator()(const KPoint& k_point) const {
    size_t hash = k_point.x;
    hash ^= k_point.y + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= k_point.z + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};
