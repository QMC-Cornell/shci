#pragma once

#include <hps/src/hps.h>
#include <set>
#include "diff_result.h"

class HalfDet {
 public:
  HalfDet(const unsigned n_elecs_hf = 0) : n_elecs_hf(n_elecs_hf) {}

  std::vector<unsigned> get_occupied_orbs() const;

  HalfDet& set(const unsigned orb);

  HalfDet& unset(const unsigned orb);

  bool has(const unsigned orb) const;

  DiffResult diff(const HalfDet& det) const;

  std::string to_string() const;

  template <class B>
  void serialize(hps::OutputBuffer<B>& buf) const;

  template <class B>
  void parse(hps::InputBuffer<B>& buf);

 private:
  unsigned n_elecs_hf;

  std::set<unsigned> orbs;

  friend bool operator==(const HalfDet& a, const HalfDet& b);

  friend bool operator!=(const HalfDet& a, const HalfDet& b);

  friend bool operator<(const HalfDet& a, const HalfDet& b);

  friend bool operator>(const HalfDet& a, const HalfDet& b);
};

template <class B>
void HalfDet::serialize(hps::OutputBuffer<B>& buf) const {
  hps::Serializer<unsigned, B>::serialize(n_elecs_hf, buf);
  unsigned level = 0;
  std::vector<unsigned> diffs;
  for (const unsigned orb : orbs) {
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
  hps::Serializer<std::vector<unsigned>, B>::serialize(diffs, buf);
}

template <class B>
void HalfDet::parse(hps::InputBuffer<B>& buf) {
  hps::Serializer<unsigned, B>::parse(n_elecs_hf, buf);
  std::vector<unsigned> diffs;
  hps::Serializer<std::vector<unsigned>, B>::parse(diffs, buf);
  orbs.clear();
  unsigned level = 0;
  for (const unsigned diff : diffs) {
    if (diff < n_elecs_hf) {
      while (level < diff) {
        orbs.insert(level);
        level++;
      }
      level++;
    } else {
      while (level < n_elecs_hf) {
        orbs.insert(level);
        level++;
      }
      orbs.insert(diff);
    }
  }
  while (level < n_elecs_hf) {
    orbs.insert(level);
    level++;
  }
}

namespace hps {
template <class B>
class Serializer<HalfDet, B> {
 public:
  static void serialize(const HalfDet& det, OutputBuffer<B>& buf) { det.serialize(buf); }
  static void parse(HalfDet& det, InputBuffer<B>& buf) { det.parse(buf); }
};
}  // namespace hps
