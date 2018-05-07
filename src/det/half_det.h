#pragma once

#include <hps/src/hps.h>
#include <set>
#include <vector>
#include "diff_result.h"

class HalfDet {
 public:
  std::vector<unsigned> get_occupied_orbs() const;

  HalfDet& set(const unsigned orb);

  HalfDet& unset(const unsigned orb);

  bool has(const unsigned orb) const;

  DiffResult diff(const HalfDet& det) const;

  std::string to_string() const;

  template <class B>
  void serialize(B& buf) const;

  template <class B>
  void parse(B& buf);

#ifndef DEBUG
 private:
#endif

#ifdef LARGE_BASIS
  std::set<unsigned> occ_orbs;
#else
  std::vector<bool> orbs;
#endif

  friend bool operator==(const HalfDet& a, const HalfDet& b);

  friend bool operator!=(const HalfDet& a, const HalfDet& b);

  friend bool operator<(const HalfDet& a, const HalfDet& b);

  friend bool operator>(const HalfDet& a, const HalfDet& b);
};

template <class B>
void HalfDet::serialize(B& buf) const {
#ifndef LARGE_BASIS
  const auto& occ_orbs = get_occupied_orbs();
#endif
  unsigned n_elecs_hf = occ_orbs.size();
  unsigned level = 0;
  std::vector<unsigned> diffs;
  for (const unsigned orb : occ_orbs) {
    if (orb < n_elecs_hf) {
      while (level < orb) {
        diffs.push_back(level);
        level++;
      }
      level++;
    } else {
      while (level < n_elecs_hf) {
        diffs.push_back(level);
        level++;
      }
      diffs.push_back(orb);
    }
  }
  while (level < n_elecs_hf) {
    diffs.push_back(level);
    level++;
  }
  buf << n_elecs_hf << diffs;
}

#ifndef LARGE_BASIS

template <class B>
void HalfDet::parse(B& buf) {
  unsigned n_elecs_hf;
  std::vector<unsigned> diffs;
  buf >> n_elecs_hf >> diffs;
  std::vector<unsigned> occ_orbs;
  orbs.clear();
  if (n_elecs_hf == 0) return;
  occ_orbs.reserve(n_elecs_hf);
  unsigned level = 0;
  for (const unsigned diff : diffs) {
    if (diff < n_elecs_hf) {
      while (level < diff) {
        occ_orbs.push_back(level);
        level++;
      }
      level++;
    } else {
      while (level < n_elecs_hf) {
        occ_orbs.push_back(level);
        level++;
      }
      occ_orbs.push_back(diff);
    }
  }
  while (level < n_elecs_hf) {
    occ_orbs.push_back(level);
    level++;
  }
  if (orbs.size() <= occ_orbs.back()) {
    orbs.assign(occ_orbs.back() + 1, false);
  }
  for (const auto orb : occ_orbs) {
    orbs[orb] = true;
  }
}

#else

template <class B>
void HalfDet::parse(B& buf) {
  unsigned n_elecs_hf;
  std::vector<unsigned> diffs;
  buf >> n_elecs_hf >> diffs;
  occ_orbs.clear();
  unsigned level = 0;
  for (const unsigned diff : diffs) {
    if (diff < n_elecs_hf) {
      while (level < diff) {
        occ_orbs.insert(level);
        level++;
      }
      level++;
    } else {
      while (level < n_elecs_hf) {
        occ_orbs.insert(level);
        level++;
      }
      occ_orbs.insert(diff);
    }
  }
  while (level < n_elecs_hf) {
    occ_orbs.insert(level);
    level++;
  }
}

#endif
