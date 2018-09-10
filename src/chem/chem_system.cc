#include "chem_system.h"

#include <fgpl/src/concurrent_hash_map.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "dooh_util.h"
#include "rdm.h"

void ChemSystem::setup() {
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);

  point_group = get_point_group(Config::get<std::string>("chem/point_group"));
  product_table.set_point_group(point_group);
  time_sym = Config::get<bool>("time_sym", false);
  has_double_excitation = Config::get<bool>("has_double_excitation", true);

  Timer::start("load integrals");
  integrals.load();
  n_orbs = integrals.n_orbs;
  orb_sym = integrals.orb_sym;
  Timer::end();

  Timer::start("setup hci queue");
  setup_hci_queue();
  Timer::end();

  dets.push_back(integrals.det_hf);
  coefs.push_back(1.0);
  energy_hf = get_hamiltonian_elem(integrals.det_hf, integrals.det_hf, 0);
  if (Parallel::is_master()) {
    printf("HF energy: " ENERGY_FORMAT "\n", energy_hf);
  }
}

void ChemSystem::setup_hci_queue() {
  sym_orbs.resize(product_table.get_n_syms() + 1);  // Symmetry starts from 1.
  for (unsigned orb = 0; orb < n_orbs; orb++) {
    unsigned sym = orb_sym[orb];
    if (sym >= sym_orbs.size()) sym_orbs.resize(sym + 1);  // For Dooh.
    sym_orbs[sym].push_back(orb);
  }
  size_t n_entries = 0;
  max_hci_queue_elem = 0.0;
  const int n_threads = Parallel::get_n_threads();
  std::vector<size_t> n_entries_local(n_threads, 0);
  std::vector<double> max_hci_queue_elem_local(n_threads, 0.0);

  // Same spin.
  hci_queue.resize(Integrals::combine2(n_orbs, 2 * n_orbs));
#pragma omp parallel for schedule(dynamic, 5)
  for (unsigned p = 0; p < n_orbs; p++) {
    const int thread_id = omp_get_thread_num();
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = p + 1; q < n_orbs; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) sym_r = DoohUtil::get_inverse(sym_r);
        sym_r = product_table.get_product(sym_q, sym_r);
        if (sym_r >= sym_orbs.size()) continue;
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
        n_entries_local[thread_id] += hci_queue.at(pq).size();
        max_hci_queue_elem_local[thread_id] =
            std::max(max_hci_queue_elem_local[thread_id], hci_queue.at(pq).front().H);
      }
    }
  }

// Opposite spin.
#pragma omp parallel for schedule(dynamic, 5)
  for (unsigned p = 0; p < n_orbs; p++) {
    const int thread_id = omp_get_thread_num();
    const unsigned sym_p = orb_sym[p];
    for (unsigned q = n_orbs + p; q < n_orbs * 2; q++) {
      const size_t pq = Integrals::combine2(p, q);
      const unsigned sym_q = product_table.get_product(sym_p, orb_sym[q - n_orbs]);
      for (unsigned r = 0; r < n_orbs; r++) {
        unsigned sym_r = orb_sym[r];
        if (point_group == PointGroup::Dooh) sym_r = DoohUtil::get_inverse(sym_r);
        sym_r = product_table.get_product(sym_q, sym_r);
        if (sym_r >= sym_orbs.size()) continue;
        for (const unsigned s : sym_orbs[sym_r]) {
          const double H = get_hci_queue_elem(p, q, r, s + n_orbs);
          if (H == 0.0) continue;
          hci_queue.at(pq).push_back(Hrs(H, r, s + n_orbs));
        }
      }
      if (hci_queue.at(pq).size() > 0) {
        std::sort(hci_queue.at(pq).begin(), hci_queue.at(pq).end(), [](const Hrs& a, const Hrs& b) {
          return a.H > b.H;
        });
        n_entries_local[thread_id] += hci_queue.at(pq).size();
        max_hci_queue_elem_local[thread_id] =
            std::max(max_hci_queue_elem_local[thread_id], hci_queue.at(pq).front().H);
      }
    }
  }

  for (int i = 0; i < n_threads; i++) {
    n_entries += n_entries_local[i];
    max_hci_queue_elem = std::max(max_hci_queue_elem, max_hci_queue_elem_local[i]);
  }

  const int proc_id = Parallel::get_proc_id();
  if (proc_id == 0) {
    printf("Max hci queue elem: " ENERGY_FORMAT "\n", max_hci_queue_elem);
    printf("Number of entries in hci queue: %'zu\n", n_entries);
  }
}

PointGroup ChemSystem::get_point_group(const std::string& str) const {
  if (Util::str_equals_ci("C1", str)) {
    return PointGroup::C1;
  } else if (Util::str_equals_ci("C2", str)) {
    return PointGroup::C2;
  } else if (Util::str_equals_ci("Cs", str)) {
    return PointGroup::Cs;
  } else if (Util::str_equals_ci("Ci", str)) {
    return PointGroup::Ci;
  } else if (Util::str_equals_ci("C2v", str)) {
    return PointGroup::C2v;
  } else if (Util::str_equals_ci("C2h", str)) {
    return PointGroup::C2h;
  } else if (Util::str_equals_ci("Coov", str) || Util::str_equals_ci("Civ", str)) {
    return PointGroup::Dooh;
  } else if (Util::str_equals_ci("D2", str)) {
    return PointGroup::D2;
  } else if (Util::str_equals_ci("D2h", str)) {
    return PointGroup::D2h;
  } else if (Util::str_equals_ci("Dooh", str) || Util::str_equals_ci("Dih", str)) {
    return PointGroup::Dooh;
  }
  throw new std::runtime_error("No point group provided");
}

double ChemSystem::get_hci_queue_elem(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) {
  if (p == q || r == s || p == r || q == s || p == s || q == r) return 0.0;
  DiffResult diff_up;
  DiffResult diff_dn;
  if (p < n_orbs && q < n_orbs) {
    assert(r < n_orbs);
    assert(s < n_orbs);
    diff_up.left_only[0] = p;
    diff_up.left_only[1] = q;
    diff_up.right_only[0] = r;
    diff_up.right_only[1] = s;
    diff_up.n_diffs = 2;
  } else if (p < n_orbs && q >= n_orbs) {
    assert(r < n_orbs);
    assert(s >= n_orbs);
    diff_up.left_only[0] = p;
    diff_dn.left_only[0] = q - n_orbs;
    diff_up.right_only[0] = r;
    diff_dn.right_only[0] = s - n_orbs;
    diff_up.n_diffs = 1;
    diff_dn.n_diffs = 1;
  } else {
    throw std::runtime_error("impossible pqrs for getting hci queue elem");
  }
  return std::abs(get_two_body_double(diff_up, diff_dn));
}

void ChemSystem::find_connected_dets(
    const Det& det,
    const double eps_max,
    const double eps_min,
    const std::function<void(const Det&, const int n_excite)>& handler) const {
  if (eps_max < eps_min) return;

  auto occ_orbs_up = det.up.get_occupied_orbs();
  auto occ_orbs_dn = det.dn.get_occupied_orbs();

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
        handler(connected_det, 1);
      } else {
        connected_det.dn.unset(p).set(r);
        handler(connected_det, 1);
      }
    }
  }

  // Add double excitations.
  if (!has_double_excitation) return;
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
      const unsigned pq = Integrals::combine2(p2, q2);
      for (const auto& hrs : hci_queue.at(pq)) {
        const double H = hrs.H;
        if (H < eps_min) break;
        if (H >= eps_max) continue;
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
        Det connected_det(det);
        p < n_orbs ? connected_det.up.unset(p) : connected_det.dn.unset(p - n_orbs);
        q < n_orbs ? connected_det.up.unset(q) : connected_det.dn.unset(q - n_orbs);
        r < n_orbs ? connected_det.up.set(r) : connected_det.dn.set(r - n_orbs);
        s < n_orbs ? connected_det.up.set(s) : connected_det.dn.set(s - n_orbs);
        handler(connected_det, 2);
      }
    }
  }
}

double ChemSystem::get_hamiltonian_elem(
    const Det& det_i, const Det& det_j, const int n_excite) const {
  return get_hamiltonian_elem_no_time_sym(det_i, det_j, n_excite);
}

double ChemSystem::get_hamiltonian_elem_no_time_sym(
    const Det& det_i, const Det& det_j, int n_excite) const {
  DiffResult diff_up;
  DiffResult diff_dn;
  if (n_excite < 0) {
    diff_up = det_i.up.diff(det_j.up);
    if (diff_up.n_diffs > 2) return 0.0;
    diff_dn = det_i.dn.diff(det_j.dn);
    n_excite = diff_up.n_diffs + diff_dn.n_diffs;
    if (n_excite > 2) return 0.0;
  } else if (n_excite > 0) {
    diff_up = det_i.up.diff(det_j.up);
    if (diff_up.n_diffs < static_cast<unsigned>(n_excite)) {
      diff_dn = det_i.dn.diff(det_j.dn);
    }
  }

  if (n_excite == 0) {
    const double one_body_energy = get_one_body_diag(det_i);
    const double two_body_energy = get_two_body_diag(det_i);
    return one_body_energy + two_body_energy + integrals.energy_core;
  } else if (n_excite == 1) {
    const double one_body_energy = get_one_body_single(diff_up, diff_dn);
    const double two_body_energy = get_two_body_single(det_i, diff_up, diff_dn);
    return one_body_energy + two_body_energy;
  } else if (n_excite == 2) {
    return get_two_body_double(diff_up, diff_dn);
  }

  throw new std::runtime_error("Calling hamiltonian with >2 exicitation");
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

void ChemSystem::update_diag_helper() {
  const size_t n_dets = get_n_dets();
  const auto& get_two_body = [&](const HalfDet& half_det) {
    const auto& occ_orbs_up = half_det.get_occupied_orbs();
    double direct_energy = 0.0;
    double exchange_energy = 0.0;
    for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
      const unsigned orb_i = occ_orbs_up[i];
      for (unsigned j = i + 1; j < occ_orbs_up.size(); j++) {
        const unsigned orb_j = occ_orbs_up[j];
        direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
        exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
      }
    }
    return direct_energy + exchange_energy;
  };

  fgpl::ConcurrentHashMap<HalfDet, double, HalfDetHasher> parallel_helper;
#pragma omp parallel for schedule(dynamic, 5)
  for (size_t i = 0; i < n_dets; i++) {
    const Det& det = dets[i];
    if (!parallel_helper.has(det.up)) {
      const double two_body = get_two_body(det.up);
      parallel_helper.set(det.up, two_body);
    }
    if (!parallel_helper.has(det.dn)) {
      const double two_body = get_two_body(det.dn);
      parallel_helper.set(det.dn, two_body);
    }
  }

  parallel_helper.for_each_serial([&](const HalfDet& half_det, const size_t, const double value) {
    diag_helper.set(half_det, value);
  });
}

double ChemSystem::get_two_body_diag(const Det& det) const {
  const auto& occ_orbs_up = det.up.get_occupied_orbs();
  const auto& occ_orbs_dn = det.dn.get_occupied_orbs();
  double direct_energy = 0.0;
  double exchange_energy = 0.0;
  // up to up.
  if (diag_helper.has(det.up)) {
    direct_energy += diag_helper.get(det.up);
    // printf("from cache\n");
  } else {
    for (unsigned i = 0; i < occ_orbs_up.size(); i++) {
      const unsigned orb_i = occ_orbs_up[i];
      for (unsigned j = i + 1; j < occ_orbs_up.size(); j++) {
        const unsigned orb_j = occ_orbs_up[j];
        direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
        exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
      }
    }
  }
  if (det.up == det.dn) {
    direct_energy *= 2;
    exchange_energy *= 2;
  } else {
    // dn to dn.
    if (diag_helper.has(det.dn)) {
      direct_energy += diag_helper.get(det.dn);
    } else {
      for (unsigned i = 0; i < occ_orbs_dn.size(); i++) {
        const unsigned orb_i = occ_orbs_dn[i];
        for (unsigned j = i + 1; j < occ_orbs_dn.size(); j++) {
          const unsigned orb_j = occ_orbs_dn[j];
          direct_energy += integrals.get_2b(orb_i, orb_i, orb_j, orb_j);
          exchange_energy -= integrals.get_2b(orb_i, orb_j, orb_j, orb_i);
        }
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
  const bool is_up_single = diff_up.n_diffs == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.left_only[0];
  const unsigned orb_j = diff.right_only[0];
  return diff.permutation_factor * integrals.get_1b(orb_i, orb_j);
}

double ChemSystem::get_two_body_single(
    const Det& det_i, const DiffResult& diff_up, const DiffResult& diff_dn) const {
  const bool is_up_single = diff_up.n_diffs == 1;
  const auto& diff = is_up_single ? diff_up : diff_dn;
  const unsigned orb_i = diff.left_only[0];
  const unsigned orb_j = diff.right_only[0];
  const auto& same_spin_half_det = is_up_single ? det_i.up : det_i.dn;
  auto oppo_spin_half_det = is_up_single ? det_i.dn : det_i.up;
  double energy = 0.0;
  for (const unsigned orb : same_spin_half_det.get_occupied_orbs()) {
    if (orb == orb_i || orb == orb_j) continue;
    energy -= integrals.get_2b(orb_i, orb, orb, orb_j);  // Exchange.
    const double direct = integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
    if (oppo_spin_half_det.has(orb)) {
      oppo_spin_half_det.unset(orb);
      energy += 2 * direct;
    } else {
      energy += direct;
    }
  }
  for (const unsigned orb : oppo_spin_half_det.get_occupied_orbs()) {
    energy += integrals.get_2b(orb_i, orb_j, orb, orb);  // Direct.
  }
  energy *= diff.permutation_factor;
  return energy;
}

double ChemSystem::get_two_body_double(const DiffResult& diff_up, const DiffResult& diff_dn) const {
  double energy = 0.0;
  if (diff_up.n_diffs == 0) {
    if (diff_dn.n_diffs != 2) return 0.0;
    const unsigned orb_i1 = diff_dn.left_only[0];
    const unsigned orb_i2 = diff_dn.left_only[1];
    const unsigned orb_j1 = diff_dn.right_only[0];
    const unsigned orb_j2 = diff_dn.right_only[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff_dn.permutation_factor;
  } else if (diff_dn.n_diffs == 0) {
    if (diff_up.n_diffs != 2) return 0.0;
    const unsigned orb_i1 = diff_up.left_only[0];
    const unsigned orb_i2 = diff_up.left_only[1];
    const unsigned orb_j1 = diff_up.right_only[0];
    const unsigned orb_j2 = diff_up.right_only[1];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2) -
             integrals.get_2b(orb_i1, orb_j2, orb_i2, orb_j1);
    energy *= diff_up.permutation_factor;
  } else {
    if (diff_up.n_diffs != 1 || diff_dn.n_diffs != 1) return 0.0;
    const unsigned orb_i1 = diff_up.left_only[0];
    const unsigned orb_i2 = diff_dn.left_only[0];
    const unsigned orb_j1 = diff_up.right_only[0];
    const unsigned orb_j2 = diff_dn.right_only[0];
    energy = integrals.get_2b(orb_i1, orb_j1, orb_i2, orb_j2);
    energy *= diff_up.permutation_factor * diff_dn.permutation_factor;
  }
  return energy;
}

void ChemSystem::post_variation(const std::vector<std::vector<size_t>>& connections) {
  if (Config::get<bool>("2rdm", false) || Config::get<bool>("get_2rdm_csv", false)) {
    RDM rdm;
    Timer::start("get 2rdm");
    rdm.get_2rdm(dets, coefs, integrals, connections, Config::get<bool>("get_2rdm_csv", false));
    Timer::end();
  }

  bool unpacked = false;

  if (Config::get<bool>("s2", false)) {
    if (time_sym && !unpacked) {
      unpack_time_sym();
      unpacked = true;
    }
    const double s2 = get_s2();
    Result::put("s2", s2);
  }

  if (Config::get<bool>("natorb", false)) {
    if (time_sym && !unpacked) {
      unpack_time_sym();
      unpacked = true;
    }
    RDM rdm;
    rdm.get_1rdm(dets, coefs, integrals);
    Timer::checkpoint("get 1rdm");

    rdm.generate_natorb_integrals(integrals);
    Timer::checkpoint("generate natorb integrals");

    exit(0);
  }

  if (Config::get<bool>("2rdm_slow", false)) {
    if (time_sym && !unpacked) {
      unpack_time_sym();
      unpacked = true;
    }
    RDM rdm;
    Timer::start("get 2rdm (slow)");
    rdm.get_2rdm_slow(dets, coefs, integrals);
    Timer::end();
  }

  if (Config::get<bool>("get_1rdm_csv", false)) {
    if (time_sym && !unpacked) {
      unpack_time_sym();
      unpacked = true;
    }
    RDM rdm;
    Timer::start("get_1rdm");
    rdm.get_1rdm(dets, coefs, integrals, true);
    Timer::end();
  }
}

//======================================================
double ChemSystem::get_s2() const {
  // Calculates <S^2> of the variation wf.
  // s^2 = n_up -n_doub - 1/2*(n_up-n_dn) + 1/4*(n_up - n_dn)^2
  //  - sum_{p != q} c_{q,dn}^{+} c_{p,dn} c_{p,up}^{+} c_{q,up}
  //
  // Created: Y. Yao, May 2018
  //======================================================
  double s2 = 0.;

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
  }

#pragma omp parallel for reduction(+ : s2)
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    Det this_det = dets[i_det];

    const auto& occ_orbs = this_det.up.get_occupied_orbs();
    unsigned num_db_occ = 0;  // number of doubly occupied orbs
    for (unsigned i = 0; i < occ_orbs.size(); i++) {
      if (this_det.dn.has(occ_orbs[i])) num_db_occ++;
    }

    // diagonal terms
    double diag = 0.5 * n_up - num_db_occ + 0.5 * n_dn;
    diag += 0.25 * (pow(n_up, 2) + pow(n_dn, 2)) - 0.5 * n_up * n_dn;
    diag *= pow(coefs[i_det], 2);
    s2 += diag;

    // double excitations
    for (unsigned i_orb = 0; i_orb < n_orbs; i_orb++) {
      if (!this_det.dn.has(i_orb)) continue;
      if (this_det.up.has(i_orb)) continue;

      for (unsigned j_orb = i_orb + 1; j_orb < n_orbs; j_orb++) {
        if (!this_det.up.has(j_orb)) continue;
        if (this_det.dn.has(j_orb)) continue;

        Det new_det = this_det;
        new_det.up.unset(j_orb);
        new_det.up.set(i_orb);
        new_det.dn.unset(i_orb);
        new_det.dn.set(j_orb);

        double coef;
        if (det2coef.count(new_det) == 1) {
          coef = det2coef[new_det];
        } else {
          continue;
        }

        const double perm_up = this_det.up.diff(new_det.up).permutation_factor;
        const double perm_dn = this_det.dn.diff(new_det.dn).permutation_factor;
        double off_diag = -2 * coef * coefs[i_det] * perm_up * perm_dn;
        s2 += off_diag;
      }  // j_orb
    }  // i_orb
  }  // i_det

  if (Parallel::is_master()) {
    printf("s_squared: %15.10f\n", s2);
  }
  return s2;
}
