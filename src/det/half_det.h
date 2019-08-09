#pragma once

#include <hps/src/hps.h>
#include <array>
#include <set>
#include <vector>
#include "diff_result.h"

//#define INF_ORBS
#define N_CHUNKS 2

class HalfDet {
 public:
  HalfDet();

  HalfDet& set(unsigned orb);

  HalfDet& unset(unsigned orb);

  bool has(unsigned orb) const;

  std::vector<unsigned> get_occupied_orbs() const;

  unsigned n_diffs(const HalfDet& rhs) const;

  DiffResult diff(const HalfDet& rhs) const;

  unsigned bit_till(unsigned p) const;

  size_t get_hash_value() const;

  void print() const;

  template <class B>
  void serialize(B& buf) const;

  template <class B>
  void parse(B& buf);

 private:
  std::array<uint64_t, N_CHUNKS> chunks;

#ifdef INF_ORBS
  std::set<unsigned> extras;
#endif

  friend bool operator==(const HalfDet& a, const HalfDet& b);

  friend bool operator!=(const HalfDet& a, const HalfDet& b);

  friend bool operator<(const HalfDet& a, const HalfDet& b);

  friend bool operator>(const HalfDet& a, const HalfDet& b);
};

template <class B>
void HalfDet::serialize(B& buf) const {
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    buf << chunks[chunk_id];
#ifdef INF_ORBS
    buf << extras;
#endif
  }
}

template <class B>
void HalfDet::parse(B& buf) {
  for (int chunk_id = 0; chunk_id < N_CHUNKS; chunk_id++) {
    buf >> chunks[chunk_id];
#ifdef INF_ORBS
    buf >> extras;
#endif
  }
}

class HalfDetHasher {
 public:
  size_t operator()(const HalfDet& half_det) const { return half_det.get_hash_value(); }
};
