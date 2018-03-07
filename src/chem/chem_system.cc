#include "chem_system.h"

#include "../parallel.h"
#include "../timer.h"

void ChemSystem::setup() {
  n_up = Config::get<int>("n_up");
  n_dn = Config::get<int>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);

  point_group = Config::get<std::string>("chem.point_group");
  const int proc_id = Parallel::get_proc_id();

  Timer::start("load integrals");
  if (proc_id == 0) {
    integrals.load();
  }
  Parallel::broadcast_object(integrals);
  n_orbs = integrals.n_orbs;
  orb_syms = integrals.orb_syms;
  Timer::end();

  Timer::start("setup hci queue");
  setup_hci_queue();
  Timer::end();

  dets.push_back(hps::serialize_to_string(integrals.det_hf));
  coefs.push_back(1.0);
}

void ChemSystem::setup_hci_queue() {
  // Same spin.
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_syms[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const unsigned sym_q = get_product(sym_p, orb_syms[q]);
      const size_t pq = integrals.combine2(p, q);
      for (unsigned r = 0; r < n_orbs; r++) {
        // TODO.
      }
    }
  }

  // Opposite spin.
}

void ChemSystem::find_connected_dets(
    const Det&, const double, const std::function<void(const Det&)>&) {}

double ChemSystem::get_hamiltonian_elem(const Det&, const Det&) { return 0.0; }
