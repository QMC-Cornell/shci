#include "heg_system.h"

#include "../config.h"
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"

void HegSystem::setup() {
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);

  r_s = Config::get<double>("heg/r_s");
  r_cut = Config::get<double>("heg/r_cut");
  const double density = 3.0 / (4.0 * Util::PI * std::pow(r_s, 3));
  const double cell_length = std::pow((n_up + n_dn) / density, 1.0 / 3);
  k_unit = 2 * Util::PI / cell_length;
  H_unit = 1.0 / (Util::PI * cell_length);
  time_sym = Config::get<bool>("time_sym", n_up == n_dn);

  Timer::start("contruct k points");
  k_points.init(r_cut);
  n_orbs = k_points.get_n_orbs();
  if (Parallel::is_master()) {
    printf("Number of spatial orbitals: %'u\n", n_orbs);
    printf("Number of spin orbitals: %'u\n", n_orbs * 2);
  }
  Timer::end();

  Timer::start("setup hci queue");
  setup_hci_queue();
  Timer::end();

  Timer::start("setup HF");
  setup_hf();
  Timer::end();
}

void HegSystem::setup_hci_queue() {
  hci_queue.clear();
  hci_queue.clear();

  max_abs_H = 0.0;

  // Common dependencies.
  const auto& k_diffs = k_points.get_k_diffs();
  const auto& sort_comparison = [](const std::pair<KPoint, double>& a,
                                   const std::pair<KPoint, double>& b) -> bool {
    return a.second > b.second;
  };

  // Same spin.
  const double diff_max_squared = 4.0 * r_cut * r_cut;
  for (const auto& diff_pq : k_diffs) {
    for (const auto& diff_pr : k_diffs) {
      const auto& diff_sr = diff_pr + diff_pr - diff_pq;  // Momentum conservation.
      if (diff_sr == 0 || diff_sr.squared_norm() > diff_max_squared) continue;
      const auto& diff_ps = diff_pr - diff_sr;
      if (diff_ps == 0) continue;
      if (diff_pr.squared_norm() == diff_ps.squared_norm()) {
        continue;
      }
      const double abs_H = std::abs(1.0 / diff_pr.squared_norm() - 1.0 / diff_ps.squared_norm());
      if (abs_H < std::numeric_limits<double>::epsilon()) continue;
      const auto& item = std::make_pair(diff_pr, abs_H * H_unit);
      hci_queue[diff_pq].push_back(item);
    }
  }
  unsigned long long n_same_spin = 0;
  for (auto& kv : hci_queue) {
    auto& items = kv.second;
    std::stable_sort(items.begin(), items.end(), sort_comparison);
    max_abs_H = std::max(max_abs_H, items.front().second);
    n_same_spin += items.size();
  }
  if (Parallel::is_master()) {
    printf("Number of same spin hci queue items: %'llu\n", n_same_spin);
  }

  // Opposite spin.
  const auto& key = KPoint(0, 0, 0);
  for (const auto& diff_pr : k_diffs) {
    const double abs_H = 1.0 / diff_pr.squared_norm();
    if (abs_H < Util::EPS) continue;
    const auto& item = std::make_pair(diff_pr, abs_H * H_unit);
    hci_queue[key].push_back(item);
  }
  std::stable_sort(hci_queue[key].begin(), hci_queue[key].end(), sort_comparison);
  max_abs_H = std::max(max_abs_H, hci_queue[key].front().second);
  if (Parallel::is_master()) {
    printf("Number of opposite spin hci queue items: %'zu\n", hci_queue[key].size());
  }
}

void HegSystem::setup_hf() {
  Det det_hf;
  double H = 0.0;

  for (unsigned p = 0; p < n_up; p++) {
    det_hf.up.set(p);
    H += k_points[p].squared_norm() * k_unit * k_unit * 0.5;
  }
  for (unsigned p = 0; p < n_dn; p++) {
    det_hf.dn.set(p);
    H += k_points[p].squared_norm() * k_unit * k_unit * 0.5;
  }

  for (unsigned p = 0; p < n_up; p++) {
    const auto& k_p = k_points[p];
    for (unsigned q = p + 1; q < n_up; q++) {
      const auto& k_q = k_points[q];
      H -= H_unit / (k_p - k_q).squared_norm();
    }
  }
  for (unsigned p = 0; p < n_dn; p++) {
    const auto& k_p = k_points[p];
    for (unsigned q = p + 1; q < n_dn; q++) {
      const auto& k_q = k_points[q];
      H -= H_unit / (k_p - k_q).squared_norm();
    }
  }

  energy_hf = H;

  if (Parallel::is_master()) {
    printf("HF energy: " ENERGY_FORMAT "\n", energy_hf);
  }

  dets.push_back(det_hf);
  coefs.push_back(1.0);
}

void HegSystem::find_connected_dets(
    const Det& det,
    const double eps_max,
    const double eps_min,
    const std::function<void(const Det&, const int n_excite)>& handler) const {
  if (eps_max < eps_min) return;

  const auto& occ_orbs_up = det.up.get_occupied_orbs();
  const auto& occ_orbs_dn = det.dn.get_occupied_orbs();

  // Add double excitations.
  if (eps_min > max_abs_H) return;
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
      const bool same_spin = p2 < n_orbs && q2 < n_orbs;
      const auto& key = same_spin ? k_points[q2] - k_points[p2] : KPoint(0, 0, 0);
      const int qs_offset = same_spin ? 0 : n_orbs;
      printf("hci_queue size: %zu\n" , hci_queue.at(key).size());
      for (const auto& item : hci_queue.at(key)) {
        const double H = item.second;
        if (H < eps_min) break;
        if (H >= eps_max) continue;
        const auto& diff_pr = item.first;
        const int r2 = k_points.find(diff_pr + k_points[p2]);
        if (r2 < 0) continue;
        unsigned r = r2;
        const int s2 = k_points.find(k_points[p2] + k_points[q2 - qs_offset] - k_points[r]);
        if (s2 < 0) continue;
        unsigned s = s2;
        s += qs_offset;
        if (p >= n_orbs && q >= n_orbs) {
          r += n_orbs;
          s += n_orbs;
        } else if (p < n_orbs && q >= n_orbs && p > q - n_orbs) {
          const int tmp = s;
          s = r + n_orbs;
          r = tmp - n_orbs;
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
