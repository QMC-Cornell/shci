#pragma once

#include <string>
#include "../base_system.h"
#include "../config.h"

class ChemSystem : public BaseSystem {
 public:
  void setup();

  void find_connected_dets(
      const Det& det,
      const double eps,
      const std::function<void(const Det&)>& connected_det_handler);

  double get_hamiltonian_elem(const Det& det_i, const Det& det_j);

 private:
  std::string point_group;
};
