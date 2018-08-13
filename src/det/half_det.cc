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
  return chunks[chunk_id] & (1ull << (orb & 0x3full));
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

size_t HalfDet::get_hash_value() const {
  size_t hash = 0;
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    hash += chunks[chunk_id];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

void HalfDet::print() const {
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    printf("%#010lx ", chunks[chunk_id]);
  }
  printf("\n");
}

unsigned HalfDet::n_diffs(const HalfDet& rhs) const {
  unsigned n_diffs = 0;
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    uint64_t chunk_left = chunks[chunk_id];
    uint64_t chunk_right = rhs.chunks[chunk_id];
    if (chunk_left == chunk_right) continue;
    uint64_t chunk_left_only = chunk_left & (~chunk_right);
    n_diffs += Util::popcnt(chunk_left_only);
  }
  return n_diffs;
}

DiffResult HalfDet::diff(const HalfDet& rhs) const {
  DiffResult res;
  unsigned n_left_only = 0;
  unsigned n_right_only = 0;
  unsigned n_elecs_left = 0;
  unsigned n_elecs_right = 0;
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
  if ((permutation_factor_helper & 1) != 0) {
    res.permutation_factor = -1;
  }
  return res;
}

unsigned HalfDet::bit_till(unsigned p) const {
  // count the total number of set orbitals below orbital p

  unsigned counter = 0;
  const auto which_chunk = p >> 6;  // orb / 64;

  for (unsigned chunk_id = 0; chunk_id < which_chunk; chunk_id++) {
    counter += Util::popcnt(chunks[chunk_id]);
  }
  counter += Util::popcnt(((1ull << (p & 63)) - 1ull) & chunks[which_chunk]);

  return counter;
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
    if (a.chunks[chunk_id] > b.chunks[chunk_id]) return false;
  }
  return false;
}

bool operator>(const HalfDet& a, const HalfDet& b) {
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    if (a.chunks[chunk_id] > b.chunks[chunk_id]) return true;
    if (a.chunks[chunk_id] < b.chunks[chunk_id]) return false;
  }
  return false;
}
