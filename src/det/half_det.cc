#include "half_det.h"

#include "../util.h"

HalfDet::HalfDet() {
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    chunks[chunk_id] = 0;
  }
}

HalfDet& HalfDet::set(unsigned orb) {
  const auto chunk_id = orb >> 6;  // orb / 64;
  chunks[chunk_id] |= 1ull << (orb & 0x3full);
  return *this;
}

HalfDet& HalfDet::unset(unsigned orb) {
  const auto chunk_id = orb >> 6;  // orb / 64;
  chunks[chunk_id] &= ~(1ull << (orb & 0x3full));
  return *this;
}

bool HalfDet::has(unsigned orb) const {
  const auto chunk_id = orb >> 6;  // orb / 64;
  return chunks[chunk_id] & (1ull << orb & 0x3full);
}

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> res;
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    uint64_t chunk = chunks[chunk_id];
    while (chunk != 0) {
      const auto tz = Util::ctz(chunk);
      chunk &= ~(1ull << tz);
      res.push_back(tz + (chunk_id << 6));
    }
  }
  return res;
}

// DiffResult HalfDet::diff(const HalfDet& det) const {
//   // assert(n_elecs_hf == det.n_elecs_hf);
//   DiffResult res;
//   unsigned n_elecs_left = 0;
//   unsigned n_elecs_right = 0;
// #ifndef LARGE_BASIS
//   const auto& occ_orbs = get_occupied_orbs();
// #endif
//   const auto& det_occ_orbs = det.get_occupied_orbs();
//   auto orbs_it_left = occ_orbs.begin();
//   auto orbs_it_right = det_occ_orbs.begin();
//   unsigned permutation_factor_helper = 0;
//   while (orbs_it_left != occ_orbs.end() && orbs_it_right != det_occ_orbs.end()) {
//     if (*orbs_it_left < *orbs_it_right) {
//       res.leftOnly.push_back(*orbs_it_left);
//       permutation_factor_helper += n_elecs_left;
//       orbs_it_left++;
//       n_elecs_left++;
//     } else if (*orbs_it_left > *orbs_it_right) {
//       res.rightOnly.push_back(*orbs_it_right);
//       permutation_factor_helper += n_elecs_right;
//       orbs_it_right++;
//       n_elecs_right++;
//     } else {
//       orbs_it_left++;
//       orbs_it_right++;
//       n_elecs_left++;
//       n_elecs_right++;
//     }
//   }
//   while (orbs_it_left != occ_orbs.end()) {
//     res.leftOnly.push_back(*orbs_it_left);
//     permutation_factor_helper += n_elecs_left;
//     orbs_it_left++;
//     n_elecs_left++;
//   }
//   while (orbs_it_right != det_occ_orbs.end()) {
//     res.rightOnly.push_back(*orbs_it_right);
//     permutation_factor_helper += n_elecs_right;
//     orbs_it_right++;
//     n_elecs_right++;
//   }
//   res.permutation_factor = (permutation_factor_helper & 1) == 1 ? -1 : 1;
//   return res;
// }

DiffResult HalfDet::diff(const HalfDet& rhs) const {
  DiffResult res;
  unsigned n_left_only = 0;
  unsigned n_right_only = 0;
  unsigned n_elecs_left = 0;
  unsigned n_elecs_right = 0;
  int tz_left;
  int tz_right;
  unsigned permutation_factor_helper = 0;
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    uint64_t chunk_left = chunks[chunk_id];
    uint64_t chunk_right = rhs.chunks[chunk_id];
    if (chunk_left == chunk_right) {
      const unsigned n_elecs = Util::popcnt(chunk_left);
      n_elecs_left += n_elecs;
      n_elecs_right += n_elecs;
      continue;
    }

    uint64_t chunk_left_only = chunk_left & (~chunk_right);
    while (chunk_left_only != 0) {
      if (n_left_only >= 2) {
        res.n_diffs = 3;
        return res;
      }
      const auto tz = Util::ctz(chunk_left_only);
      const uint64_t bit = 1ull << tz;
      chunk_left_only &= ~bit;
      res.left_only[n_left_only] = tz + (chunk_id << 6);
      n_left_only++;
      permutation_factor_helper += n_elecs_left + Util::popcnt(chunk_left & (bit - 1));
    }

    uint64_t chunk_right_only = chunk_right & (~chunk_left);
    while (chunk_right_only != 0) {
      if (n_right_only >= 2) {
        res.n_diffs = 3;
        return res;
      }
      const auto tz = Util::ctz(chunk_right_only);
      const uint64_t bit = 1ull << tz;
      chunk_right_only &= ~bit;
      res.right_only[n_right_only] = tz + (chunk_id << 6);
      n_right_only++;
      permutation_factor_helper += n_elecs_right + Util::popcnt(chunk_right & (bit - 1));
    }

    if (chunk_id != N_CHUNKS - 1) {
      n_elecs_left += Util::popcnt(chunk_left);
      n_elecs_right += Util::popcnt(chunk_right);
    }
  }
  res.n_diffs = n_left_only;
  if (permutation_factor_helper & 1 != 0) {
    res.permutation_factor = -1;
  }
  return res;
}

bool operator==(const HalfDet& a, const HalfDet& b) {
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    if (a.chunks[chunk_id] != b.chunks[chunk_id]) return false;
  }
  return true;
}

bool operator!=(const HalfDet& a, const HalfDet& b) { return !(a == b); }

bool operator<(const HalfDet& a, const HalfDet& b) {
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    if (a.chunks[chunk_id] < b.chunks[chunk_id]) return true;
  }
  return false;
}

bool operator>(const HalfDet& a, const HalfDet& b) { return !(a < b); }
