#pragma once

#include <string>
#include <vector>

class BaseSystem {
 public:
  int n_up;

  int n_dn;

  int n_elecs;

  std::vector<std::string> dets;

  std::vector<double> coefs;

  void setup();

  void setup_variation(){};

  void setup_perturbation(){};
};
