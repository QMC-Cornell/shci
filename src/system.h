#pragma once

#include <string>
#include <vector>

class System {
 public:
  std::vector<std::string> dets;
  std::vector<double> coefs;

  virtual ~System() = default;
};
