#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"

class ChemSystem : public BaseSystem {
 public:
  void setup();

 private:
  std::string point_group;
};
