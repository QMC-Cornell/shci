#include "half_det.h"

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> occupied_orbs;
  occupied_orbs.reserve(n_elecs_hf);
  unsigned level = 0;
  for (unsigned orb : orbs_from) {
    while (level < orb) occupied_orbs.push_back(level++);
    level++;
  }
  while (level < n_elecs_hf) occupied_orbs.push_back(level++);
  for (unsigned orb : orbs_to) occupied_orbs.push_back(orb);
  return occupied_orbs;
}

void HalfDet::set(const unsigned orb) {
  if (orb < n_elecs_hf) {
    orbs_from.erase(orb);
  } else {
    orbs_to.insert(orb);
  }
}

void HalfDet::unset(const unsigned orb) {
  if (orb < n_elecs_hf) {
    orbs_from.insert(orb);
  } else {
    orbs_to.erase(orb);
  }
}

std::vector<unsigned> HalfDet::get_diff_orbs(const HalfDet& det) const {
  assert(n_elecs_hf == det.n_elecs_hf);
  auto diff_orbs = get_set_diff(orbs_from, det.orbs_from);
  const auto& orbs_to_diff = get_set_diff(orbs_to, det.orbs_to);
  diff_orbs.reserve(diff_orbs.size() + orbs_to_diff.size());
  diff_orbs.insert(diff_orbs.end(), orbs_to_diff.begin(), orbs_to_diff.end());
  return diff_orbs;
}

std::vector<unsigned> HalfDet::get_set_diff(
    const std::set<unsigned>& a, const std::set<unsigned>& b) const {
  std::vector<unsigned> set_diff;
  auto a_it = a.begin();
  auto b_it = b.begin();
  while (a_it != a.end() && b_it != b.end()) {
    while (*a_it < *b_it) {
      set_diff.push_back(*a_it);
      a_it++;
    }
    while (*a_it > *b_it) {
      set_diff.push_back(*b_it);
      b_it++;
    }
    while (*a_it == *b_it) {
      a_it++;
      b_it++;
    }
  }
  while (a_it != a.end()) set_diff.push_back(*a_it);
  while (b_it != b.end()) set_diff.push_back(*b_it);
  return set_diff;
}
