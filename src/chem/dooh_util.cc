#include "dooh_util.h"

#include <cstdlib>

int DoohUtil::get_lz(int ind, int& gu) {
  if (ind <= 2) {
    gu = ind - 1;
    return 0;
  } else {
    int lz = 1 + ((ind - 3) >> 2);
    if ((((ind - 1) >> 1) & 1) == 0) lz = -lz;
    gu = (ind + 1) & 1;
    return lz;
  }
}

int DoohUtil::get_ind(int lz, int gu) {
  if (lz == 0) {
    return gu + 1;
  } else {
    int ind = (std::abs(lz) << 2) - 1 + gu;
    if (lz < 0) ind += 2;
    return ind;
  }
}

int DoohUtil::get_product(int i, int j) {
  int gu_i = 0;
  int gu_j = 0;
  const int lz_i = get_lz(i, gu_i);
  const int lz_j = get_lz(j, gu_j);
  return get_ind(lz_i + lz_j, (gu_i + gu_j) & 1);
}

int DoohUtil::get_inverse(int i) {
  int gu;
  const int lz = get_lz(i, gu);
  if (lz > 0) {
    return i + 2;
  } else if (lz < 0) {
    return i - 2;
  } else {
    return i;
  }
}
