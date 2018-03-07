#pragma once

#include "point_group.h"

class ProductTable {
 public:
  unsigned get_product(const unsigned a, const unsigned b) const {}

  void set_point_group(const PointGroup point_group);

 private:
  PointGroup point_group;
};
