#include "half_det.h"

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> occupied_orbs;
  occupied_orbs.reserve(n_elecs_hf);
  for (unsigned orb : orbs) {
    occupied_orbs.push_back(orb);
  }
  return occupied_orbs;
}

HalfDet& HalfDet::set(const unsigned orb) {
  orbs.insert(orb);
  return *this;
}

HalfDet& HalfDet::unset(const unsigned orb) {
  orbs.erase(orb);
  return *this;
}

bool HalfDet::has(const unsigned orb) const { return orbs.count(orb) == 1; }

DiffResult HalfDet::diff(const HalfDet& det) const {
  assert(n_elecs_hf == det.n_elecs_hf);
  DiffResult res;
  size_t n_elecs_left = 0;
  size_t n_elecs_right = 0;
  auto orbs_it_left = orbs.begin();
  auto orbs_it_right = det.orbs.begin();
  int permutation_factor_helper = 0;
  while (orbs_it_left != orbs.end() && orbs_it_right != det.orbs.end()) {
    if (*orbs_it_left < *orbs_it_right) {
      res.leftOnly.push_back(*orbs_it_left);
      permutation_factor_helper += n_elecs_left;
      orbs_it_left++;
      n_elecs_left++;
    } else if (*orbs_it_left > *orbs_it_right) {
      res.rightOnly.push_back(*orbs_it_right);
      permutation_factor_helper += n_elecs_right;
      orbs_it_right++;
      n_elecs_right++;
    } else {
      orbs_it_left++;
      orbs_it_right++;
      n_elecs_left++;
      n_elecs_right++;
    }
  }
  while (orbs_it_left != orbs.end()) {
    res.leftOnly.push_back(*orbs_it_left);
    permutation_factor_helper += n_elecs_left;
    orbs_it_left++;
    n_elecs_left++;
  }
  while (orbs_it_right != det.orbs.end()) {
    res.rightOnly.push_back(*orbs_it_right);
    permutation_factor_helper += n_elecs_right;
    orbs_it_right++;
    n_elecs_right++;
  }
  res.permutation_factor = (permutation_factor_helper & 1) == 1 ? -1 : 1;
  return res;
}

bool operator==(const HalfDet& a, const HalfDet& b) {
  return a.n_elecs_hf == b.n_elecs_hf && a.orbs == b.orbs;
}

bool operator!=(const HalfDet& a, const HalfDet& b) { return !(a == b); }

bool operator<(const HalfDet& a, const HalfDet& b) {
  if (a.n_elecs_hf != b.n_elecs_hf) {
    throw std::runtime_error("cannot compare half det of different HF elecs.");
  }
  return a.orbs < b.orbs;
}

bool operator>(const HalfDet& a, const HalfDet& b) { return !(a < b); }
