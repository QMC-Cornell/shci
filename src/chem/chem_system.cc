#include "chem_system.h"

#include <cfloat>
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

void ChemSystem::setup_hci_queue() { sym_orbs.resize(product_table.get_n_syms() + 1);  // Symmetry starts from 1.
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
          n_entries++;
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
  Det det_pq;
  Det det_rs;
  if (p < n_orbs && q < n_orbs) {
    assert(r < n_orbs);
    assert(s < n_orbs);
    det_pq.up.set(p);
    det_pq.up.set(q);
    det_rs.up.set(r);
    det_rs.up.set(s);
  } else if (p < n_orbs && q >= n_orbs) {
    assert(r < n_orbs);
    assert(s >= n_orbs);
    det_pq.up.set(p);
    det_pq.dn.set(q - n_orbs);
    det_rs.up.set(r);
    det_rs.dn.set(s - n_orbs);
  } else {
    throw std::runtime_error("impossible pqrs for getting hci queue elem");
  }
  return abs(get_two_body_double(det_pq, det_rs, true));
}

void ChemSystem::find_connected_dets(
    const Det& det,
    const double eps_max_in,
    const double eps_min_in,
    const std::function<void(const Det&, const double)>& connected_det_handler) const {
  const double eps_max = time_sym ? eps_max_in * Util::SQRT2 : eps_max_in;
  const double eps_min = time_sym ? eps_min_in * Util::SQRT2 : eps_min_in;
  if (eps_max < eps_min) return;

  const auto& occ_orbs_up = det.up.get_occupied_orbs();
  const auto& occ_orbs_dn = det.dn.get_occupied_orbs();

  const auto& prospective_det_handler = [&](Det& connected_det, const unsigned n_excite) {
    if (time_sym) {
      if (connected_det.up == connected_det.dn && z < 0) return;
      if (connected_det.up == det.dn && connected_det.dn == det.up) return;
    }
    double matrix_elem = get_hamiltonian_elem(det, connected_det, n_excite);
    if (abs(matrix_elem) > eps_max || abs(matrix_elem) < eps_min) return;
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
      Det connected_det(det);
      if (p_id < n_up) {
        connected_det.up.unset(p).set(r);
      } else {
        connected_det.dn.unset(p).set(r);
      }
      prospective_det_handler(connected_det, 1);
    }
  }

  // Add double excitations.
  if (eps_min > max_hci_queue_elem) return;
  for (unsigned p_id = 0; p_id < n_elecs; p_id++) {
    for (unsigned q_id = p_id; q_id < n_elecs; q_id++) {
      const unsigned p = p_id < n_up ? occ_orbs_up[p_id] : occ_orbs_dn[p_id - n_up] + n_orbs;
      const unsigned q = q_id < n_up ? occ_orbs_up[q_id] : occ_orbs_dn[q_id - n_up] + n_orbs;
      double p2 = p;
      double q2 = q;
      if (p > n_orbs && q > n_orbs) {
        p2 -= n_orbs;
        q2 -= n_orbs;
      } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
        p2 = q - n_orbs;
        q2 = p + n_orbs;
      }
      const unsigned pq = Integrals::combine2(p, q);
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
          s = r - n_orbs;
          r = tmp_r;
        }
        const bool occ_r = r < n_orbs ? det.up.has(r) : det.dn.has(r - n_orbs);
        if (occ_r) continue;
        const bool occ_s = s < n_orbs ? det.up.has(s) : det.dn.has(s - n_orbs);
        if (occ_s) continue;
        Det connected_det(det);
        p < n_orbs ? connected_det.up.unset(p) : connected_det.dn.unset(p - n_orbs);
        q < n_orbs ? connected_det.up.unset(q) : connected_det.dn.unset(q - n_orbs);
        r < n_orbs ? connected_det.up.set(r) : connected_det.dn.set(r - n_orbs);
        s < n_orbs ? connected_det.up.set(s) : connected_det.dn.set(s - n_orbs);
        prospective_det_handler(connected_det, 2);
      }
    }
  }
  
  
}

double ChemSystem::get_hamiltonian_elem(
    const Det& det_i, const Det& det_j, const unsigned n_excite) const { 
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
    const Det& det_i, const Det& det_j, const unsigned n_excite) const {
  if (n_excite == 0) {
    const double one_body_energy = get_one_body_diag(det_i);
    const double two_body_energy = get_two_body_diag(det_i);
    return one_body_energy + two_body_energy + integrals.energy_core;
  } else if (n_excite == 1) {
    const double one_body_energy = get_one_body_single(det_i, det_j);
    const double two_body_energy = get_two_body_single(det_i, det_j);
    return one_body_energy + two_body_energy;
  } else if (n_excite == 2) {
    return get_two_body_double(det_i, det_j);
  }
  throw new std::runtime_error("Calling hamiltonian kernel with >2 exicitation");
}

double ChemSystem::get_one_body_diag(const Det& det) const {
  return 0.0;
}

double ChemSystem::get_two_body_diag(const Det& det) const {
  return 0.0;
}

double ChemSystem::get_one_body_single(const Det& det_i, const Det& det_j) const {
  return 0.0;
}

double ChemSystem::get_two_body_single(const Det& det_i, const Det& det_j) const {
  return 0.0;
}

double ChemSystem::get_two_body_double(
    const Det& det_i, const Det& det_j, const bool no_sign) const {
  double two_body_energy = 0.0;
  if (det_i.up == det_j.up) {
    const auto& diff_ij = det_i.dn.diff(det_j.dn);
    if (diff_ij.first.size() != 2 || diff_ij.second.size() != 2) return 0.0;
    two_body_energy =
        integrals.get_2b(diff_ij.first[0], diff_ij.second[0], diff_ij.first[1], diff_ij.second[1]) -
        integrals.get_2b(diff_ij.first[0], diff_ij.second[1], diff_ij.first[1], diff_ij.second[0]);
  } else if (det_i.dn == det_j.dn) {
    const auto& diff_ij = det_i.up.diff(det_j.up);
    if (diff_ij.first.size() != 2 || diff_ij.second.size() != 2) return 0.0;
    two_body_energy =
        integrals.get_2b(diff_ij.first[0], diff_ij.second[0], diff_ij.first[1], diff_ij.second[1]) -
        integrals.get_2b(diff_ij.first[0], diff_ij.second[1], diff_ij.first[1], diff_ij.second[0]);
  } else {
    const auto& diff_ij_up = det_i.up.diff(det_j.up);
    const auto& diff_ij_dn = det_i.dn.diff(det_j.dn);
    if (diff_ij_up.first.size() != 1 || diff_ij_up.second.size() != 1) return 0.0;
    if (diff_ij_dn.first.size() != 1 || diff_ij_dn.second.size() != 1) return 0.0;
    two_body_energy = integrals.get_2b(
        diff_ij_up.first[0], diff_ij_up.second[0], diff_ij_dn.first[0], diff_ij_dn.second[0]);
  }

  if (!no_sign) {
    // Get permutation factor.
  }

  return two_body_energy;
}
