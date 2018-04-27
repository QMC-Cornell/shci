#include "chem_system.h"

#include <cfloat>
#include <cmath>
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"

void ChemSystem::setup() {
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);
  time_sym = Config::get<bool>("time_sym", false);
  z = Config::get<int>("z", 1);

  point_group = get_point_group(Config::get<std::string>("chem/point_group"));
  product_table.set_point_group(point_group);

  Timer::start("load integrals");
  integrals.load();
  n_orbs = integrals.n_orbs;
  orb_sym = integrals.orb_sym;
  Timer::end();

  Timer::start("setup hci queue");
  setup_hci_queue();
  Timer::end();

  dets.push_back(hps::serialize_to_string(integrals.det_hf));
  coefs.push_back(1.0);
  energy_hf = get_hamiltonian_elem(integrals.det_hf, integrals.det_hf, 0);
  if (Parallel::is_master()) {
    printf("HF energy: " ENERGY_FORMAT "\n", energy_hf);
  }
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
  sym_orbs.resize(product_table.get_n_syms() + 1);  // Symmetry starts from 1.
  for (unsigned orb = 0; orb < n_orbs; orb++) sym_orbs[orb_sym[orb]].push_back(orb);
  size_t n_entries = 0;
  max_hci_queue_elem = 0.0;

  // Same spin.
  hci_queue.resize(Integrals::combine2(n_orbs, 2 * n_orbs));
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) exit(0);  // TODO: dih inv.
        sym_r = product_table.get_product(sym_q, sym_r);
        for (const unsigned s : sym_orbs[sym_r]) {
          if (s < r) continue;
          const double H = get_hci_queue_elem(p, q, r, s);
          if (H == 0.0) continue;
          hci_queue.at(pq).push_back(Hrs(H, r, s));
        }
      }
      if (hci_queue.at(pq).size() > 0) {
        std::sort(hci_queue.at(pq).begin(), hci_queue.at(pq).end(), [](const Hrs& a, const Hrs& b) {
          return a.H > b.H;
        });
        n_entries += hci_queue.at(pq).size();
        max_hci_queue_elem = std::max(max_hci_queue_elem, hci_queue.at(pq).front().H);
      }
    }
  }

  // Opposite spin.
  unsigned cnt = 0;
  for (unsigned p = 0; p < n_orbs; p++) {
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = n_orbs + p; q < n_orbs * 2; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q - n_orbs]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) exit(0);  // TODO: dih inv.
        sym_r = product_table.get_product(sym_q, sym_r);
        for (const unsigned s : sym_orbs[sym_r]) {
          const double H = get_hci_queue_elem(p, q, r, s + n_orbs);
          if (H == 0.0) continue;
          hci_queue.at(pq).push_back(Hrs(H, r, s + n_orbs));
          max_hci_queue_elem = std::max(max_hci_queue_elem, H);
        }
      }
      if (hci_queue.at(pq).size() > 0) {
        std::sort(hci_queue.at(pq).begin(), hci_queue.at(pq).end(), [](const Hrs& a, const Hrs& b) {
          return a.H > b.H;
        });
        n_entries += hci_queue.at(pq).size();
        max_hci_queue_elem = std::max(max_hci_queue_elem, hci_queue.at(pq).front().H);
      }
      cnt += hci_queue.at(pq).size();
    }
  }

  const int proc_id = Parallel::get_proc_id();
  if (proc_id == 0) {
    printf("Max hci queue elem: " ENERGY_FORMAT "\n", max_hci_queue_elem);
    printf("Number of entries in hci queue: %'zu\n", n_entries);
  }
}

double ChemSystem::get_hci_queue_elem(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) {
  if (p == q || r == s || p == r || q == s || p == s || q == r) return 0.0;
  DiffResult diff_up;
  DiffResult diff_dn;
  if (p < n_orbs && q < n_orbs) {
    assert(r < n_orbs);
    assert(s < n_orbs);
    diff_up.leftOnly.push_back(p);
    diff_up.leftOnly.push_back(q);
    diff_up.rightOnly.push_back(r);
    diff_up.rightOnly.push_back(s);
  } else if (p < n_orbs && q >= n_orbs) {
    assert(r < n_orbs);
    assert(s >= n_orbs);
    diff_up.leftOnly.push_back(p);
    diff_dn.leftOnly.push_back(q - n_orbs);
    diff_up.rightOnly.push_back(r);
    diff_dn.rightOnly.push_back(s - n_orbs);
  } else {
    throw std::runtime_error("impossible pqrs for getting hci queue elem");
  }
  return std::abs(get_two_body_double(diff_up, diff_dn));
}

void ChemSystem::find_connected_dets(
    const Det& det,
    const double eps_max_in,
    const double eps_min_in,
    const std::function<void(const Det&, const double)>& connected_det_handler) const {
  const double eps_max = time_sym ? eps_max_in * Util::SQRT2 : eps_max_in;
  const double eps_min = time_sym ? eps_min_in * Util::SQRT2 : eps_min_in;
  if (eps_max < eps_min) return;

  auto occ_orbs_up = det.up.get_occupied_orbs();
  auto occ_orbs_dn = det.dn.get_occupied_orbs();
  size_t n_con = 0;

  const auto& prospective_det_handler = [&](Det& connected_det, const unsigned n_excite) {
    if (time_sym) {
      if (connected_det.up == connected_det.dn && z < 0) return;
      if (connected_det.up == det.dn && connected_det.dn == det.up) return;
    }
    double matrix_elem = get_hamiltonian_elem(det, connected_det, n_excite);
    if (std::abs(matrix_elem) > eps_max || std::abs(matrix_elem) < eps_min) return;
    if (time_sym) {
      if (det.up == det.dn && connected_det.up != connected_det.dn) {
        matrix_elem *= Util::SQRT2_INV;
      } else if (det.up != det.dn && connected_det.up == connected_det.dn) {
        matrix_elem *= Util::SQRT2;
      }
      if (connected_det.up > connected_det.dn) {
        HalfDet tmp_half_det = connected_det.up;
        connected_det.up = connected_det.dn;
        connected_det.dn = tmp_half_det;
        matrix_elem *= z;
      }
    }
    connected_det_handler(connected_det, matrix_elem);
    n_con++;
  };

  // Add single excitations.
  for (unsigned p_id = 0; p_id < n_elecs; p_id++) {
    const unsigned p = p_id < n_up ? occ_orbs_up[p_id] : occ_orbs_dn[p_id - n_up];
    const unsigned sym_p = orb_sym[p];
    for (unsigned r = 0; r < n_orbs; r++) {
      if (p_id < n_up) {
        if (det.up.has(r)) continue;
      } else {
        if (det.dn.has(r)) continue;
      }
      if (orb_sym[r] != sym_p) continue;
      Det connected_det = det;
      if (p_id < n_up) {
        connected_det.up.unset(p).set(r);
      } else {
        connected_det.dn.unset(p).set(r);
      }
      const size_t n_con_prev = n_con;
      prospective_det_handler(connected_det, 1);
      if (n_con > n_con_prev) {
        // printf("p r: %u %u\n", p, r);
      }
    }
  }

  // Add double excitations.
  if (eps_min > max_hci_queue_elem) return;
  for (unsigned p_id = 0; p_id < n_elecs; p_id++) {
    for (unsigned q_id = p_id + 1; q_id < n_elecs; q_id++) {
      const unsigned p = p_id < n_up ? occ_orbs_up[p_id] : occ_orbs_dn[p_id - n_up] + n_orbs;
      const unsigned q = q_id < n_up ? occ_orbs_up[q_id] : occ_orbs_dn[q_id - n_up] + n_orbs;
      double p2 = p;
      double q2 = q;
      if (p >= n_orbs && q >= n_orbs) {
        p2 -= n_orbs;
        q2 -= n_orbs;
      } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
        p2 = q - n_orbs;
        q2 = p + n_orbs;
      }
      size_t pq_count = 0;
      const unsigned pq = Integrals::combine2(p2, q2);
      for (const auto& hrs : hci_queue.at(pq)) {
        const double H = hrs.H;
        if (H < eps_min) break;
        if (H > eps_max) continue;
        unsigned r = hrs.r;
        unsigned s = hrs.s;
        if (p >= n_orbs && q >= n_orbs) {
          r += n_orbs;
          s += n_orbs;
        } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
          const unsigned tmp_r = s - n_orbs;
          s = r + n_orbs;
          r = tmp_r;
        }
        const bool occ_r = r < n_orbs ? det.up.has(r) : det.dn.has(r - n_orbs);
        if (occ_r) continue;
        const bool occ_s = s < n_orbs ? det.up.has(s) : det.dn.has(s - n_orbs);
        if (occ_s) continue;
        Det connected_det = det;
        p < n_orbs ? connected_det.up.unset(p) : connected_det.dn.unset(p - n_orbs);
        q < n_orbs ? connected_det.up.unset(q) : connected_det.dn.unset(q - n_orbs);
        r < n_orbs ? connected_det.up.set(r) : connected_det.dn.set(r - n_orbs);
        s < n_orbs ? connected_det.up.set(s) : connected_det.dn.set(s - n_orbs);
        const size_t n_con_prev = n_con;
        prospective_det_handler(connected_det, 2);
        if (n_con > n_con_prev) {
          pq_count++;
          // printf("p q r s: %u %u %u %u\n", p, q, r, s);
        }
      }
      // printf("p q count: %u %u %zu\n", p, q, pq_count);
    }
    // printf("n_cons: %zu\n", n_con);
    // exit(0);
  }

  // printf("n_cons: %zu\n", n_con);
  // exit(0);
}

double ChemSystem::get_hamiltonian_elem(
    const Det& det_i, const Det& det_j, const int n_excite) const {
  if (!time_sym) return get_hamiltonian_elem_kernel(det_i, det_j, n_excite);
  double norm_ket_inv = 1.0;
  double norm_bra = 1.0;
  bool check = true;
  if (det_j.up == det_j.dn) norm_ket_inv = Util::SQRT2_INV;
  if (det_i.up == det_i.dn) {
    norm_bra = Util::SQRT2;
    check = false;
  }
  const double matrix_elem_1 = get_hamiltonian_elem_kernel(det_i, det_j, n_excite);
  double matrix_elem_2 = 1.0;
  if (check) {
    if (det_i.up != det_j.dn) {
      matrix_elem_2 = get_hamiltonian_elem_kernel(det_i, det_j, n_excite);
    } else {
      matrix_elem_2 = matrix_elem_1;
    }
  }
  return norm_bra * norm_ket_inv * (matrix_elem_1 + z * matrix_elem_2);
}

double ChemSystem::get_hamiltonian_elem_kernel(
    const Det& det_i, const Det& det_j, int n_excite) const {
  DiffResult diff_up;
  DiffResult diff_dn;
  if (n_excite < 0) {
    diff_up = det_i.up.diff(det_j.up);
    const int n_excite_up = diff_up.leftOnly.size();
    if (n_excite_up > 2) return 0.0;
    diff_dn = det_i.dn.diff(det_j.dn);
    const int n_excite_dn = diff_dn.leftOnly.size();
    n_excite = n_excite_up + n_excite_dn;
    if (n_excite > 2) return 0.0;
  } else {
    diff_up = det_i.up.diff(det_j.up);
    const int n_excite_up = diff_up.leftOnly.size();
    if (n_excite_up < n_excite) {
      diff_dn = det_i.dn.diff(det_j.dn);
    }
  }

  if (n_excite == 0) {
    const double one_body_energy = get_one_body_diag(det_i);
    const double two_body_energy = get_two_body_diag(det_i);
    // printf("1 2 body core diag: %f %f %f\n", one_body_energy, two_body_energy,
    // integrals.energy_core);
    return one_body_energy + two_body_energy + integrals.energy_core;
  } else if (n_excite == 1) {
    const double one_body_energy = get_one_body_single(diff_up, diff_dn);
    const double two_body_energy = get_two_body_single(det_i, diff_up, diff_dn);
    // printf("1 2 body single: %f %f %f\n", one_body_energy, two_body_energy, one_body_energy +
    // two_body_energy);
    return one_body_energy + two_body_energy;
  } else if (n_excite == 2) {
    return get_two_body_double(diff_up, diff_dn);
  }

  throw new std::runtime_error("Calling hamiltonian kernel with >2 exicitation");
}

double ChemSystem::get_one_body_diag(const Det& det) const {
  double energy = 0.0;
  for (const auto& orb : det.up.get_occupied_orbs()) {
    energy += integrals.get_1b(orb, orb);
  }
  if (det.up == det.dn) {
    energy *= 2;
  } else {
    for (const auto& orb : det.dn.get_occupied_orbs()) {
      energy += integrals.get_1b(orb, orb);
    }
  }
  return energy;
}

double ChemSystem::get_two_body_diag(const Det& det) const {
  const auto& occ_orbs_up = det.up.get_occupied_orbs();
  const auto& occ_orbs_dn = det.dn.get_occupied_orbs();
  double direct_energy = 0.0;
  double exchange_energy = 0.0;
  // up to up.
  for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
    const unsigned orb_i = occ_orbs_up[i];
    for (unsigned j = i + 1; j < occ_orbs_up.size(); j++) {
      const unsigned orb_j = occ_orbs_up[j];
      direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
      exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
    }
  }
  if (det.up == det.dn) {
    direct_energy *= 2;
    exchange_energy *= 2;
  } else {
    // dn to dn.
    for (unsigned i = 0; i < occ_orbs_dn.size(); i++) {
      const unsigned orb_i = occ_orbs_dn[i];
      for (unsigned j = i + 1; j < occ_orbs_dn.size(); j++) {
        const unsigned orb_j = occ_orbs_dn[j];
        direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
        exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
      }
    }
  }
  // up to dn.
  for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
    const unsigned orb_i = occ_orbs_up[i];
    for (unsigned j = 0; j < occ_orbs_dn.size(); j++) {
      const unsigned orb_j = occ_orbs_dn[j];
      direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
    }
  }
  return direct_energy + exchange_energy;
}

double ChemSystem::get_one_body_single(const DiffResult& diff_up, const DiffResult& diff_dn) const {
  const bool is_up_single = diff_up.leftOnly.size() == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.leftOnly[0];
  const unsigned orb_j = diff.rightOnly[0];
  return diff.permutation_factor * integrals.get_1b(orb_i, orb_j);
}

double ChemSystem::get_two_body_single(
    const Det& det_i, const DiffResult& diff_up, const DiffResult& diff_dn) const {
  const bool is_up_single = diff_up.leftOnly.size() == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.leftOnly[0];
  const unsigned orb_j = diff.rightOnly[0];
  const auto& same_spin_half_det = is_up_single ? det_i.up : det_i.dn;
  const auto& oppo_spin_half_det = is_up_single ? det_i.dn : det_i.up;
  double energy = 0.0;
  for (const unsigned orb : same_spin_half_det.get_occupied_orbs()) {
    if (orb == orb_i || orb == orb_j) continue;
    energy -= integrals.get_2b(orb_i, orb, orb, orb_j);  // Exchange.
    energy += integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
  }
  for (const unsigned orb : oppo_spin_half_det.get_occupied_orbs()) {
    energy += integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
  }
  energy *= diff.permutation_factor;
  return energy;
}

double ChemSystem::get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const {
  double energy = 0.0;
  if (diff_up.leftOnly.size() == 0) {
    const auto& diff = diff_dn;
    if (diff.leftOnly.size() != 2 || diff.rightOnly.size() != 2) return 0.0;
    const unsigned orb_i1 = diff.leftOnly[0];
    const unsigned orb_i2 = diff.leftOnly[1];
    const unsigned orb_j1 = diff.rightOnly[0];
    const unsigned orb_j2 = diff.rightOnly[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff.permutation_factor;
  } else if (diff_dn.leftOnly.size() == 0) {
    const auto& diff = diff_up;
    if (diff.leftOnly.size() != 2 || diff.rightOnly.size() != 2) return 0.0;
    const unsigned orb_i1 = diff.leftOnly[0];
    const unsigned orb_i2 = diff.leftOnly[1];
    const unsigned orb_j1 = diff.rightOnly[0];
    const unsigned orb_j2 = diff.rightOnly[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff.permutation_factor;
  } else {
    if (diff_up.leftOnly.size() != 1 || diff_up.rightOnly.size() != 1) return 0.0;
    if (diff_dn.leftOnly.size() != 1 || diff_dn.rightOnly.size() != 1) return 0.0;
    const unsigned orb_i1 = diff_up.leftOnly[0];
    const unsigned orb_i2 = diff_dn.leftOnly[0];
    const unsigned orb_j1 = diff_up.rightOnly[0];
    const unsigned orb_j2 = diff_dn.rightOnly[0];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2);
    energy *= diff_up.permutation_factor * diff_dn.permutation_factor;
  }
  return energy;
}
