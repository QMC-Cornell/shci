#include "k_point.h"

size_t KPoint::squared_norm() const {
  size_t result = 0;
  result += static_cast<size_t>(x) * x;
  result += static_cast<size_t>(y) * y;
  result += static_cast<size_t>(z) * z;
  return result;
}

KPoint operator-(const KPoint& lhs, const KPoint& rhs) {
  KPoint result;
  result.x = lhs.x - rhs.x;
  result.y = lhs.y - rhs.y;
  result.z = lhs.z - rhs.z;
  return result;
}

KPoint operator+(const KPoint& lhs, const KPoint& rhs) {
  KPoint result;
  result.x = lhs.x + rhs.x;
  result.y = lhs.y + rhs.y;
  result.z = lhs.z + rhs.z;
  return result;
}

bool operator==(const KPoint& lhs, const KPoint& rhs) {
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

bool operator==(const KPoint& lhs, const int rhs) {
  return (lhs.x == rhs) && (lhs.y == rhs) && (lhs.z == rhs);
}
