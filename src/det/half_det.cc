#include "half_det.h"

#include "../util.h"

HalfDet::HalfDet() {
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    chunks[chunk_id] = 0;
  }
}

HalfDet& HalfDet::set(unsigned orb) {
  if (orb >= N_CHUNKS << 6) {
#ifdef INF_ORBS
    extras.insert(orb);
#else
    throw std::invalid_argument("n orbs > n trunks * 64");
#endif
  } else {
    const auto chunk_id = orb >> 6;  // orb / 64;
    chunks[chunk_id] |= 1ull << (orb & 0x3full);
  }
  return *this;
}

HalfDet& HalfDet::unset(unsigned orb) {
  if (orb >= N_CHUNKS << 6) {
#ifdef INF_ORBS
    extras.erase(orb);
#else
    throw std::invalid_argument("n orbs > n trunks * 64");
#endif
  } else {
    const auto chunk_id = orb >> 6;  // orb / 64;
    chunks[chunk_id] &= ~(1ull << (orb & 0x3full));
  }
  return *this;
}

bool HalfDet::has(unsigned orb) const {
  if (orb >= N_CHUNKS << 6) {
#ifdef INF_ORBS
    return extras.count(orb) == 1;
#else
    throw std::invalid_argument("n orbs > n trunks * 64");
#endif
  } else {
    const auto chunk_id = orb >> 6;  // orb / 64;
    return chunks[chunk_id] & (1ull << (orb & 0x3full));
  }
}

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> res;
  res.reserve(16);
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    uint64_t chunk = chunks[chunk_id];
    while (chunk != 0) {
      const auto tz = Util::ctz(chunk);
      chunk &= ~(1ull << tz);
      res.push_back(tz + (chunk_id << 6));
    }
  }
#ifdef INF_ORBS
  for (unsigned orb : extras) {
    res.push_back(orb);
  }
#endif
  return res;
}

size_t HalfDet::get_hash_value() const {
  size_t hash = 0;
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    hash += chunks[chunk_id];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
#ifdef INF_ORBS
  for (unsigned orb : extras) {
    hash += orb;
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
#endif
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

void HalfDet::print() const {
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    printf("%#010lx ", chunks[chunk_id]);
  }
#ifdef INF_ORBS
  for (unsigned orb : extras) {
    printf("%u ", orb);
  }
#endif
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
#ifdef INF_ORBS
  for (unsigned orb : extras) {
    if (rhs.extras.count(orb) == 0) n_diffs++;
  }
#endif
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
      const unsigned n_elecs_trunk = Util::popcnt(chunk_left);
      n_elecs_left += n_elecs_trunk;
      n_elecs_right += n_elecs_trunk;
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

    n_elecs_left += Util::popcnt(chunk_left);
    n_elecs_right += Util::popcnt(chunk_right);
  }

#ifdef INF_ORBS
  for (const unsigned orb : extras) {
    if (rhs.extras.count(orb) == 1) {
      n_elecs_left++;
      continue;
    }
    if (n_left_only >= 2) {
      res.n_diffs = 3;
      return res;
    }
    res.left_only[n_left_only] = orb;
    n_left_only++;
    permutation_factor_helper += n_elecs_left;
    n_elecs_left++;
  }

  for (const unsigned orb : rhs.extras) {
    if (extras.count(orb) == 1) {
      n_elecs_right++;
      continue;
    }
    if (n_right_only >= 2) {
      res.n_diffs = 3;
      return res;
    }
    res.right_only[n_right_only] = orb;
    n_right_only++;
    permutation_factor_helper += n_elecs_right;
    n_elecs_right++;
  }
#endif

  res.n_diffs = n_left_only;
  if ((permutation_factor_helper & 1) != 0) {
    res.permutation_factor = -1;
  }
  return res;
}

unsigned HalfDet::bit_till(unsigned p) const {
  // WARNING: Not working for n_elecs > N_CHUNKS * 64
  // count the total number of set orbitals below orbital p
  assert(p < N_CHUNKS << 6);
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
#ifdef INF_ORBS
  return a.extras == b.extras;
#else
  return true;
#endif
}

bool operator!=(const HalfDet& a, const HalfDet& b) { return !(a == b); }

bool operator<(const HalfDet& a, const HalfDet& b) {
#ifdef INF_ORBS
  auto it_a = a.extras.rbegin();
  auto it_b = b.extras.rbegin();
  while (it_a != a.extras.rend() && it_b != b.extras.rend()) {
    if (*it_a < *it_b) {
      return true;
    } else if (*it_a > *it_b) {
      return false;
    }
    it_a++;
    it_b++;
  }
  if (it_a != a.extras.rend() && it_b == b.extras.rend()) {
    return false;
  }
  if (it_a == a.extras.rend() && it_b != b.extras.rend()) {
    return true;
  }
#endif
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    if (a.chunks[chunk_id] < b.chunks[chunk_id]) return true;
    if (a.chunks[chunk_id] > b.chunks[chunk_id]) return false;
  }
  return false;
}

bool operator>(const HalfDet& a, const HalfDet& b) {
#ifdef INF_ORBS
  auto it_a = a.extras.rbegin();
  auto it_b = b.extras.rbegin();
  while (it_a != a.extras.rend() && it_b != b.extras.rend()) {
    if (*it_a < *it_b) {
      return false;
    } else if (*it_a > *it_b) {
      return true;
    }
    it_a++;
    it_b++;
  }
  if (it_a != a.extras.rend() && it_b == b.extras.rend()) {
    return true;
  }
  if (it_a == a.extras.rend() && it_b != b.extras.rend()) {
    return false;
  }
#endif
  for (int chunk_id = N_CHUNKS - 1; chunk_id >= 0; chunk_id--) {
    if (a.chunks[chunk_id] > b.chunks[chunk_id]) return true;
    if (a.chunks[chunk_id] < b.chunks[chunk_id]) return false;
  }
  return false;
}
