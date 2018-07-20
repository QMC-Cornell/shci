#pragma once

#include <vector>
#include "point_group.h"

class ProductTable {
 public:
  ProductTable() : point_group() {}

  unsigned get_product(const unsigned a, const unsigned b) const;

  unsigned get_n_syms() const;

  void set_point_group(const PointGroup point_group);

 private:
  PointGroup point_group;

  std::vector<std::vector<unsigned>> product_table_elems;

  template <unsigned N>
  void set_table_elems(const unsigned table[][N]);
};

template <unsigned N>
void ProductTable::set_table_elems(const unsigned table[][N]) {
  product_table_elems.resize(N);
  for (unsigned i = 0; i < N; i++) {
    product_table_elems[i].resize(N);
    for (unsigned j = 0; j < N; j++) product_table_elems[i][j] = table[i][j];
  }
}
