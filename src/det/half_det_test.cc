#include "half_det.h"
#include <gtest/gtest.h>

TEST(HalfDetTest, SetAndGetOrbitals) {
  HalfDet::n_orbs = 3;
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

TEST(HalfDetTest, SerializeAndParse) {
  HalfDet::n_orbs = 3;
  HalfDet half_det;
  std::string serialized = hps::to_string(half_det);
  EXPECT_EQ(half_det, hps::from_string<HalfDet>(serialized));

  half_det.set(0);
  half_det.set(1);
  serialized = hps::to_string(half_det);
  EXPECT_EQ(half_det, hps::from_string<HalfDet>(serialized));
  half_det.unset(0);
  half_det.set(5);
  serialized = hps::to_string(half_det);
  EXPECT_EQ(half_det, hps::from_string<HalfDet>(serialized));

  half_det.unset(2);
  half_det.set(6);
  EXPECT_NE(half_det, hps::from_string<HalfDet>(serialized));
  serialized = hps::to_string(half_det);
  EXPECT_EQ(half_det, hps::from_string<HalfDet>(serialized));
}

TEST(HalfDetTest, Diff) {
  HalfDet::n_orbs = 3;
  HalfDet a;
  HalfDet b;
  a.set(0);
  b.set(1);
  auto diff = a.diff(b);
  EXPECT_EQ(diff.leftOnly.size(), 1);
  EXPECT_EQ(diff.leftOnly[0], 0);
  EXPECT_EQ(diff.rightOnly.size(), 1);
  EXPECT_EQ(diff.rightOnly[0], 1);

  a.set(1);
  b.set(3);
  diff = a.diff(b);
  EXPECT_EQ(diff.leftOnly.size(), 1);
  EXPECT_EQ(diff.leftOnly[0], 0);
  EXPECT_EQ(diff.rightOnly.size(), 1);
  EXPECT_EQ(diff.rightOnly[0], 3);
}
