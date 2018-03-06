#include "chem_system.h"

#include "../parallel.h"
#include "../timer.h"

void ChemSystem::setup() {
  BaseSystem::setup();
  dets.clear();
  coefs.clear();
  point_group = Config::get<std::string>("chem.point_group");
  const int proc_id = Parallel::get_proc_id();

  Timer::start("load integrals");
  if (proc_id == 0) {
    integrals.load();
  }
  // Parallel::broadcast_object(integrals);
  Timer::end();
}

void ChemSystem::find_connected_dets(
    const Det&, const double, const std::function<void(const Det&)>&) {}

double ChemSystem::get_hamiltonian_elem(const Det&, const Det&) { return 0.0; }
