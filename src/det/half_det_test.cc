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
  half_det.set(3);
  const auto& orbs_2 = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs_2.size(), 2);
}

TEST(HalfDetTest, SetAndGetOrbitalsLarge) {
  HalfDet half_det;
  EXPECT_FALSE(half_det.has(0));
  EXPECT_EQ(half_det.get_occupied_orbs().size(), 0);
  half_det.set(130);
  EXPECT_TRUE(half_det.has(130));
  const auto& orbs = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs[0], 130);
  half_det.set(129);
  const auto& orbs_2 = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs_2.size(), 2);
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

TEST(HalfDetTest, DiffLarge) {
  HalfDet a;
  HalfDet b;
  a.set(0);
  b.set(130);
  auto diff = a.diff(b);
  EXPECT_EQ(diff.n_diffs, 1);
  EXPECT_EQ(diff.left_only[0], 0);
  EXPECT_EQ(diff.right_only[0], 130);

  a.set(130);
  a.set(256);
  b.set(128);
  b.set(129);
  diff = a.diff(b);
  EXPECT_EQ(diff.n_diffs, 2);
  EXPECT_EQ(diff.left_only[0], 0);
  EXPECT_EQ(diff.left_only[1], 256);
  EXPECT_EQ(diff.right_only[0], 128);
  EXPECT_EQ(diff.right_only[1], 129);
}
