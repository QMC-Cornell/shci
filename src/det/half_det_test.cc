#include "half_det.h"
#include <gtest/gtest.h>

TEST(HalfDetTest, SetAndGetOrbitals) {
  HalfDet half_det;
  EXPECT_FALSE(half_det.has(0));
  EXPECT_EQ(half_det.get_occupied_orbs().size(), 0);
  half_det.set(0);
  EXPECT_TRUE(half_det.has(0));
  const auto& orbs = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs[0], 0);
  const auto& orbs_2 = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs_2.size(), 1);
}

TEST(HalfDetTest, Diff) {
  HalfDet a;
  HalfDet b;
  a.set(0);
  b.set(1);
  auto diff = a.diff(b);
  EXPECT_EQ(diff.n_diffs, 1);
  EXPECT_EQ(diff.left_only[0], 0);
  EXPECT_EQ(diff.right_only[0], 1);

  a.set(1);
  b.set(3);
  diff = a.diff(b);
  EXPECT_EQ(diff.n_diffs, 1);
  EXPECT_EQ(diff.left_only[0], 0);
  EXPECT_EQ(diff.right_only[0], 3);
}
