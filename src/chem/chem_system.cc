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
  orb_sym = integrals.orb_sym;
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
  // Orbs by sym.
  // sym_n_orbs.assign(product_table.get_n_syms() + 1, 0);
  sym_orbs.resize(product_table.get_n_syms());
  for (const auto sym : orb_sym) {
    // sym_n_orbs[sym]++;
    sym_orbs.push_back(sym);
  }

  // Same spin.
  hci_queue.reserve(Integrals::combine2(n_orbs, 2 * n_orbs));
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q]);
      for (unsigned r = 0; r < n_orbs; r++) {
        sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) exit(0);  // TODO: dih inv.
        sym_r = product_table.get_product(sym_q, sym_r);
        for (const unsigned s : sym_orbs[sym_r]) {
          if (s < r) continue;
          const double H = get_hci_queue_elem(p, q, r, s);
          if (H == 0.0) continue;
          hci_queue[pq].push_back(RSH(r, s, H));
        }
      }
      const size_t pq = Integrals::combine2(p, q);
      std::sort(hci_queue[pq].begin(), hci_queue[pq].end(), [](const RSH a, const RSH b) {
        return a.H > b.H;
      });
    }
  }

  // Opposite spin.
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = n_orbs + p; q < n_orbs * 2; q++) {
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q - n_orbs]);
      for (unsigned r = 0; r < n_orbs; r++) {
        sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) exit(0);  // TODO: dih inv.
        sym_r = product_table.get_product(sym_q, sym_r);
        for (const unsigned s : sym_orbs[sym_r]) {
          const double H = get_hci_queue_elem(p, q, r, s + n_orbs);
          if (H == 0.0) continue;
          hci_queue[pq].push_back(RSH(r, s + n_orbs, H));
        }
      }
      const size_t pq = Integrals::combine2(p, q);
      std::sort(hci_queue[pq].begin(), hci_queue[pq].end(), [](const RSH a, const RSH b) {
        return a.H > b.H;
      });
    }
  }
}

double ChemSystem::get_hci_queue_elem(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) {
  if (p == q || r == s || p == r || q == s || p == s || q == r) return 0.0;
  Det det_pq;
  Det det_rs;
  if (p < n_orbs && q < n_orbs) {
    det_pq.up.set(p);
    det_pq.up.set(q);
    det_rs.up.set(r);
    det_rs.up.set(s);
  } else if (p < n_orbs && q >= n_orbs) {
    det_pq.up.set(p);
    det_pq.dn.set(q - n_orbs);
    det_rs.up.set(r);
    det_rs.dn.set(s - n_orbs);
  } else {
    throw std::runtime_error("impossible pqrs for getting hci queue elem");
  }
  return std::abs(get_two_body_double(det_pq, det_rs, true));
}

void ChemSystem::find_connected_dets(
    const Det&, const double, const std::function<void(const Det&)>&) {}

double ChemSystem::get_hamiltonian_elem(const Det&, const Det&) { return 0.0; }

double ChemSystem::get_two_body_double(const Det& det_i, const Det& det_j, const bool no_sign) {
  return 0.0;
}
