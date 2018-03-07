#include "chem_system.h"

#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"

void ChemSystem::setup() {
  n_up = Config::get<int>("n_up");
  n_dn = Config::get<int>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);

  point_group = get_point_group(Config::get<std::string>("chem.point_group"));
  product_table.set_point_group(point_group);
  const int proc_id = Parallel::get_proc_id();

  Timer::start("load integrals");
  if (proc_id == 0) integrals.load();
  // Parallel::broadcast_object(integrals);
  n_orbs = integrals.n_orbs;
  orb_syms = integrals.orb_syms;
  Timer::end();

  Timer::start("setup hci queue");
  if (proc_id == 0) setup_hci_queue();
  Timer::end();

  dets.push_back(hps::serialize_to_string(integrals.det_hf));
  coefs.push_back(1.0);
}

PointGroup ChemSystem::get_point_group(const std::string& str) const {
  if (Util::str_iequals("D2h", str)) {
    return PointGroup::D2h;
  } else if (Util::str_iequals("Dooh", str) || Util::str_iequals("Dih", str)) {
    return PointGroup::Dooh;
  }
  return PointGroup::None;
}

void ChemSystem::setup_hci_queue() {
  // Same spin.
  hci_queue.reserve(Integrals::combine2(n_orbs, 2 * n_orbs));
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_syms[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const unsigned sym_q = product_table.get_product(sym_p, orb_syms[q]);
      const size_t pq = Integrals::combine2(p, q);
      for (unsigned r = 0; r < n_orbs; r++) {
        sym_r = orb_syms[r];
        if (point_group == PointGroup::Dooh) exit(0); // TODO.
         sym_r = product_table.get_product(sym_q, sym_r);
      }
    }
  }

  // Opposite spin.
}

void ChemSystem::find_connected_dets(
    const Det&, const double, const std::function<void(const Det&)>&) {}

double ChemSystem::get_hamiltonian_elem(const Det&, const Det&) { return 0.0; }
