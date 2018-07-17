#include "product_table.h"

#include "dooh_util.h"

const static unsigned C1[][1] = {{1}};

const static unsigned CsCi[][2] = {{1, 2}, {2, 1}};

const static unsigned C2vC2h[][4] = {{1, 2, 3, 4}, {2, 1, 4, 3}, {3, 4, 1, 2}, {4, 3, 2, 1}};

const static unsigned D2h[][8] = {{1, 2, 3, 4, 5, 6, 7, 8},
                                  {2, 1, 4, 3, 6, 5, 8, 7},
                                  {3, 4, 1, 2, 7, 8, 5, 6},
                                  {4, 3, 2, 1, 8, 7, 6, 5},
                                  {5, 6, 7, 8, 1, 2, 3, 4},
                                  {6, 5, 8, 7, 2, 1, 4, 3},
                                  {7, 8, 5, 6, 3, 4, 1, 2},
                                  {8, 7, 6, 5, 4, 3, 2, 1}};

void ProductTable::set_point_group(const PointGroup point_group) {
  this->point_group = point_group;
  if (point_group == PointGroup::C1) {
    set_table_elems<1>(C1);
  } else if (point_group == PointGroup::Cs || point_group == PointGroup::Ci) {
    set_table_elems<2>(CsCi);
  } else if (point_group == PointGroup::C2v || point_group == PointGroup::C2h) {
    set_table_elems<4>(C2vC2h);
  } else if (point_group == PointGroup::D2h) {
    set_table_elems<8>(D2h);
  }
};

unsigned ProductTable::get_product(const unsigned a, const unsigned b) const {
  if (point_group == PointGroup::Dooh) {
    return DoohUtil::get_product(a, b);
  }
  return product_table_elems[a - 1][b - 1];
}

unsigned ProductTable::get_n_syms() const {
  if (point_group == PointGroup::Dooh) return 128;  // Estimate.
  return product_table_elems.size();
}
