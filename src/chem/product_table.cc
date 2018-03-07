#include "product_table.h"

static unsigned D2H[][8] = {{1, 2, 3, 4, 5, 6, 7, 8},
                            {2, 1, 4, 3, 6, 5, 8, 7},
                            {3, 4, 1, 2, 7, 8, 5, 6},
                            {4, 3, 2, 1, 8, 7, 6, 5},
                            {5, 6, 7, 8, 1, 2, 3, 4},
                            {6, 5, 8, 7, 2, 1, 4, 3},
                            {7, 8, 5, 6, 3, 4, 1, 2},
                            {8, 7, 6, 5, 4, 3, 2, 1}};

void ProductTable::set_point_group(const PointGroup point_group) {
  this->point_group = point_group;
  if (point_group == PointGroup::D2h) {
    set_table_elems<8>(D2H);
  }
};

unsigned ProductTable::get_product(const unsigned a, const unsigned b) const {
  if (point_group == PointGroup::Dooh) {
    return 1;
  }
  return product_table_elems[a - 1][b - 1];
}
