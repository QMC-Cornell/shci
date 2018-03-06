#include "chem_system.h"

void ChemSystem::setup() {
  BaseSystem::setup();
  dets.clear();
  coefs.clear();
  point_group = Config::get<std::string>("chem.point_group");
  integrals.load_fcidump();
}

void ChemSystem::find_connected_dets(
    const Det&, const double, const std::function<void(const Det&)>&) {}

double ChemSystem::get_hamiltonian_elem(const Det&, const Det&) { return 0.0; }
