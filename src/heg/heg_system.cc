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
  same_spin_hci_queue.clear();
  oppo_spin_hci_queue.clear();

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
      same_spin_hci_queue[diff_pq].push_back(item);
    }
  }
  unsigned long long n_same_spin = 0;
  for (auto& kv : same_spin_hci_queue) {
    auto& items = kv.second;
    std::stable_sort(items.begin(), items.end(), sort_comparison);
    max_abs_H = std::max(max_abs_H, items.front().second);
    n_same_spin += items.size();
  }
  if (Parallel::is_master()) {
    printf("Number of same spin hci queue items: %'llu\n", n_same_spin);
  }

  // Opposite spin.
  for (const auto& diff_pr : k_diffs) {
    const double abs_H = 1.0 / diff_pr.squared_norm();
    if (abs_H < Util::EPS) continue;
    const auto& item = std::make_pair(diff_pr, abs_H * H_unit);
    oppo_spin_hci_queue.push_back(item);
  }
  std::stable_sort(oppo_spin_hci_queue.begin(), oppo_spin_hci_queue.end(), sort_comparison);
  max_abs_H = std::max(max_abs_H, oppo_spin_hci_queue.front().second);
  if (Parallel::is_master()) {
    printf("Number of opposite spin hci queue items: %'zu\n", oppo_spin_hci_queue.size());
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
