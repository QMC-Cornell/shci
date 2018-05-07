#include "half_det.h"

#ifndef LARGE_BASIS

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> res;
  for (size_t i = 0; i < orbs.size(); i++) {
    if (orbs[i]) res.push_back(i);
  }
  return res;
}

HalfDet& HalfDet::set(const unsigned orb) {
  if (orbs.size() <= orb) orbs.resize(orb + 1, false);
  orbs[orb] = true;
  return *this;
}

HalfDet& HalfDet::unset(const unsigned orb) {
  if (orbs.size() <= orb) return *this;
  orbs[orb] = false;
  return *this;
}

bool HalfDet::has(const unsigned orb) const { return orbs.size() > orb && orbs[orb]; }

std::string HalfDet::to_string() const {
  std::string res = "";
  for (size_t i = 0; i < orbs.size(); i++) {
    if (orbs[i]) {
      res = res + " " + std::to_string(i);
    }
  }
  return res;
}

DiffResult HalfDet::diff(const HalfDet& det) const {
  // assert(n_elecs_hf == det.n_elecs_hf);
  DiffResult res;
  unsigned n_elecs_left = 0;
  unsigned n_elecs_right = 0;
  auto orbs_it_left = orbs.begin();
  auto orbs_it_right = det.orbs.begin();
  unsigned permutation_factor_helper = 0;
  while (orbs_it_left != orbs.end() && orbs_it_right != det.orbs.end()) {
    if (!(*orbs_it_left) && *orbs_it_right) {
      res.leftOnly.push_back(*orbs_it_left);
      permutation_factor_helper += n_elecs_left;
    } else if (*orbs_it_left && !(*orbs_it_right)) {
      res.rightOnly.push_back(*orbs_it_right);
      permutation_factor_helper += n_elecs_right;
    }
    if (*orbs_it_left) n_elecs_left++;
    if (*orbs_it_right) n_elecs_right++;
    orbs_it_left++;
    orbs_it_right++;
  }
  while (orbs_it_left != orbs.end()) {
    if (*orbs_it_left) {
      res.leftOnly.push_back(*orbs_it_left);
      permutation_factor_helper += n_elecs_left;
      n_elecs_left++;
    }
    orbs_it_left++;
  }
  while (orbs_it_right != det.orbs.end()) {
    if (*orbs_it_right) {
      res.rightOnly.push_back(*orbs_it_right);
      permutation_factor_helper += n_elecs_right;
      n_elecs_right++;
    }
    orbs_it_right++;
  }
  res.permutation_factor = (permutation_factor_helper & 1) == 1 ? -1 : 1;
  return res;
}

bool operator==(const HalfDet& a, const HalfDet& b) {
  return a.get_occupied_orbs() == b.get_occupied_orbs();
}

bool operator!=(const HalfDet& a, const HalfDet& b) { return !(a == b); }

bool operator<(const HalfDet& a, const HalfDet& b) {
  return a.get_occupied_orbs() < b.get_occupied_orbs();
}

bool operator>(const HalfDet& a, const HalfDet& b) { return !(a < b); }

#else

std::vector<unsigned> HalfDet::get_occupied_orbs() const {
  std::vector<unsigned> res;
  res.reserve(occ_orbs.size());
  for (unsigned orb : occ_orbs) {
    res.push_back(orb);
  }
  return res;
}

HalfDet& HalfDet::set(const unsigned orb) {
  occ_orbs.insert(orb);
  return *this;
}

HalfDet& HalfDet::unset(const unsigned orb) {
  occ_orbs.erase(orb);
  return *this;
}

bool HalfDet::has(const unsigned orb) const { return occ_orbs.count(orb) == 1; }

std::string HalfDet::to_string() const {
  std::string res = "";
  for (unsigned orb : occ_orbs) res = res + " " + std::to_string(orb);
  return res;
}

DiffResult HalfDet::diff(const HalfDet& det) const {
  // assert(n_elecs_hf == det.n_elecs_hf);
  DiffResult res;
  unsigned n_elecs_left = 0;
  unsigned n_elecs_right = 0;
  auto orbs_it_left = occ_orbs.begin();
  auto orbs_it_right = det.occ_orbs.begin();
  unsigned permutation_factor_helper = 0;
  while (orbs_it_left != occ_orbs.end() && orbs_it_right != det.occ_orbs.end()) {
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
  while (orbs_it_left != occ_orbs.end()) {
    res.leftOnly.push_back(*orbs_it_left);
    permutation_factor_helper += n_elecs_left;
    orbs_it_left++;
    n_elecs_left++;
  }
  while (orbs_it_right != det.occ_orbs.end()) {
    res.rightOnly.push_back(*orbs_it_right);
    permutation_factor_helper += n_elecs_right;
    orbs_it_right++;
    n_elecs_right++;
  }
  res.permutation_factor = (permutation_factor_helper & 1) == 1 ? -1 : 1;
  return res;
}

bool operator==(const HalfDet& a, const HalfDet& b) { return a.occ_orbs == b.occ_orbs; }

bool operator!=(const HalfDet& a, const HalfDet& b) { return !(a == b); }

bool operator<(const HalfDet& a, const HalfDet& b) { return a.occ_orbs < b.occ_orbs; }

bool operator>(const HalfDet& a, const HalfDet& b) { return !(a < b); }

#endif