#include "half_det.h"
#include <gtest/gtest.h>

TEST(HalfDetTest, SetAndGetOrbitals) {
  HalfDet half_det(3);
  EXPECT_FALSE(half_det.has(0));
  EXPECT_EQ(half_det.get_occupied_orbs().size(), 0);
  half_det.set(0);
  EXPECT_TRUE(half_det.has(0));
  const auto& orbs = half_det.get_occupied_orbs();
  EXPECT_EQ(orbs.size(), 1);
  EXPECT_EQ(orbs[0], 0);
}

TEST(HalfDetTest, SerializeAndParse) {
  HalfDet half_det(3);
  std::string serialized = hps::serialize_to_string(half_det);
  EXPECT_EQ(half_det, hps::parse_from_string<HalfDet>(serialized));

  half_det.set(0);
  half_det.set(2);
  serialized = hps::serialize_to_string(half_det);
  EXPECT_EQ(half_det, hps::parse_from_string<HalfDet>(serialized));

  half_det.unset(0);
  half_det.set(5);
  serialized = hps::serialize_to_string(half_det);
  EXPECT_EQ(half_det, hps::parse_from_string<HalfDet>(serialized));

  half_det.unset(2);
  half_det.set(6);
  serialized = hps::serialize_to_string(half_det);
  EXPECT_EQ(half_det, hps::parse_from_string<HalfDet>(serialized));
}
