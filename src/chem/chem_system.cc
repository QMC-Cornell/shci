#include "chem_system.h"

void ChemSystem::setup() {
  BaseSystem::setup();
  point_group = Config::get<std::string>("chem.point_group");
}
