#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "sparse_matrix.h"

template <class S>
class Hamiltonian {
 public:
  Hamiltonian();

  SparseMatrix<double> matrix;

  void update(const S& system);

  void clear();

 private:
  bool is_master = false;

  size_t n_dets = 0;

  size_t n_dets_prev = 0;

  unsigned n_up = 0;

  unsigned n_dn = 0;

  std::vector<std::string> unique_alphas;

  std::vector<std::string> unique_betas;

  std::unordered_map<std::string, size_t> alpha_to_id;

  std::unordered_map<std::string, size_t> beta_to_id;

  std::unordered_map<std::string, std::pair<std::vector<size_t>, std::vector<size_t>>>
      abm1_to_ab_ids;

  std::vector<std::vector<size_t>> alpha_id_to_single_ids;

  std::vector<std::vector<size_t>> beta_id_to_single_ids;

  // Sorted by unique beta id.
  std::vector<std::vector<size_t>> alpha_id_to_beta_ids;

  // Sorted by unique beta id.
  std::vector<std::vector<size_t>> alpha_id_to_det_ids;

  // Sorted by unique alpha id.
  std::vector<std::vector<size_t>> beta_id_to_alpha_ids;

  // Sorted by unique alpha id.
  std::vector<std::vector<size_t>> beta_id_to_det_ids;

  // Augment unique alphas/betas and alpha/beta to det info.
  void update_abdet(const S& system);

  // Update unique alpha/beta minus one.
  void update_abm1(const S& system);

  // Update alpha/beta singles lists.
  void update_absingles(const S& system);

  void update_matrix(const S& system);

  void sort_by_first(std::vector<size_t>& vec1, std::vector<size_t>& vec2);
};

template <class S>
Hamiltonian<S>::Hamiltonian() {
  is_master = Parallel::get_proc_id() == 0;
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
}

template <class S>
void Hamiltonian<S>::update(const S& system) {
  n_dets_prev = n_dets;
  n_dets = system.get_n_dets();
  if (n_dets_prev == n_dets) return;

  update_abdet(system);
  update_abm1(system);
  Timer::checkpoint("updated abm1");
  update_absingles(system);
  Timer::checkpoint("updated absingles");
  update_matrix(system);
  Timer::checkpoint("updated hamiltonian matrix");
}

template <class S>
void Hamiltonian<S>::update_abdet(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = hps::parse_from_string<Det>(system.get_det(i));

    // Obtain alpha id.
    const auto& alpha = hps::serialize_to_string(det.up);
    size_t alpha_id;
    if (alpha_to_id.count(alpha) == 0) {
      alpha_id = alpha_to_id.size();
      alpha_to_id[alpha] = alpha_id;
      unique_alphas.push_back(alpha);
      alpha_id_to_beta_ids.resize(alpha_id + 1);
      alpha_id_to_det_ids.resize(alpha_id + 1);
    } else {
      alpha_id = alpha_to_id[alpha];
    }

    // Obtain beta id.
    const auto& beta = hps::serialize_to_string(det.dn);
    size_t beta_id;
    if (beta_to_id.count(beta) == 0) {
      beta_id = beta_to_id.size();
      beta_to_id[beta] = beta_id;
      unique_betas.push_back(beta);
      beta_id_to_alpha_ids.resize(beta_id + 1);
      beta_id_to_det_ids.resize(beta_id + 1);
    } else {
      beta_id = beta_to_id[beta];
    }

    // Update alpha/beta to det info.
    alpha_id_to_beta_ids[alpha_id].push_back(beta_id);
    alpha_id_to_det_ids[alpha_id].push_back(i);
    beta_id_to_alpha_ids[beta_id].push_back(alpha_id);
    beta_id_to_det_ids[beta_id].push_back(i);
    updated_alphas.insert(alpha_id);
    updated_betas.insert(beta_id);
  }

  // Sort updated alpha/beta to det info.
  for (const size_t alpha_id : updated_alphas) {
    sort_by_first(alpha_id_to_beta_ids[alpha_id], alpha_id_to_det_ids[alpha_id]);
  }
  for (const size_t beta_id : updated_betas) {
    sort_by_first(beta_id_to_alpha_ids[beta_id], beta_id_to_det_ids[beta_id]);
  }
}

template <class S>
void Hamiltonian<S>::update_abm1(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = hps::parse_from_string<Det>(system.get_det(i));

    // Update alpha m1.
    const auto& alpha = hps::serialize_to_string(det.up);
    const size_t alpha_id = alpha_to_id[alpha];
    if (updated_alphas.count(alpha_id) == 0) {
      const auto& up_elecs = det.up.get_occupied_orbs();
      HalfDet half_det = det.up;
      for (unsigned j = 0; j < n_up; j++) {
        half_det.unset(up_elecs[j]);
        const auto& alpha_m1 = hps::serialize_to_string(half_det);
        abm1_to_ab_ids[alpha_m1].first.push_back(alpha_id);
        half_det.set(up_elecs[j]);
      }
      updated_alphas.insert(alpha_id);
    }

    // Update beta m1.
    const auto& beta = hps::serialize_to_string(det.dn);
    const size_t beta_id = beta_to_id[beta];
    if (updated_betas.count(beta_id) == 0) {
      const auto& dn_elecs = det.dn.get_occupied_orbs();
      HalfDet half_det = det.dn;
      for (unsigned j = 0; j < n_dn; j++) {
        half_det.unset(dn_elecs[j]);
        const auto& beta_m1 = hps::serialize_to_string(half_det);
        abm1_to_ab_ids[beta_m1].second.push_back(beta_id);
        half_det.set(dn_elecs[j]);
      }
      updated_betas.insert(beta_id);
    }
  }
}

template <class S>
void Hamiltonian<S>::update_absingles(const S& system) {
  std::unordered_set<size_t> updated_alphas;
  std::unordered_set<size_t> updated_betas;
  alpha_id_to_single_ids.resize(alpha_to_id.size());
  beta_id_to_single_ids.resize(beta_to_id.size());

  for (size_t i = n_dets_prev; i < n_dets; i++) {
    const auto& det = hps::parse_from_string<Det>(system.get_det(i));

    const auto& alpha = hps::serialize_to_string(det.up);
    const size_t alpha_id = alpha_to_id[alpha];
    updated_alphas.insert(alpha_id);

    const auto& beta = hps::serialize_to_string(det.dn);
    const size_t beta_id = beta_to_id[beta];
    updated_betas.insert(beta_id);
  }

  const size_t n_unique_alphas = alpha_to_id.size();
  const size_t n_unique_betas = beta_to_id.size();

  std::vector<omp_lock_t> locks;
  const size_t n_locks = std::max(n_unique_alphas, n_unique_betas);
  locks.resize(n_locks);
  for (auto& lock : locks) omp_init_lock(&lock);

#pragma omp parallel for schedule(static, 1)
  for (size_t alpha_id = 0; alpha_id < n_unique_alphas; alpha_id++) {
    const auto& alpha = unique_alphas[alpha_id];
    HalfDet half_det = hps::parse_from_string<HalfDet>(alpha);
    const auto& up_elecs = half_det.get_occupied_orbs();
    for (unsigned j = 0; j < n_up; j++) {
      half_det.unset(up_elecs[j]);
      const auto& alpha_m1 = hps::serialize_to_string(half_det);
      if (abm1_to_ab_ids.count(alpha_m1) == 1) {
        for (const size_t alpha_single : abm1_to_ab_ids[alpha_m1].first) {
          if (alpha_single == alpha_id) continue;
          if (alpha_id > alpha_single && updated_alphas.count(alpha_id) &&
              updated_alphas.count(alpha_single)) {
            continue;  // Delegate to the alpha_single outer iteration.
          }
          omp_set_lock(&locks[alpha_id]);
          alpha_id_to_single_ids[alpha_id].push_back(alpha_single);
          omp_unset_lock(&locks[alpha_id]);
          omp_set_lock(&locks[alpha_single]);
          alpha_id_to_single_ids[alpha_single].push_back(alpha_id);
          omp_unset_lock(&locks[alpha_single]);
        }
      }
      half_det.set(up_elecs[j]);
    }
  }

#pragma omp parallel for schedule(static, 1)
  for (size_t beta_id = 0; beta_id < n_unique_betas; beta_id++) {
    const auto& beta = unique_betas[beta_id];
    HalfDet half_det = hps::parse_from_string<HalfDet>(beta);
    const auto& dn_elecs = half_det.get_occupied_orbs();
    for (unsigned j = 0; j < n_dn; j++) {
      half_det.unset(dn_elecs[j]);
      const auto& beta_m1 = hps::serialize_to_string(half_det);
      if (abm1_to_ab_ids.count(beta_m1) == 1) {
        for (const size_t beta_single : abm1_to_ab_ids[beta_m1].second) {
          if (beta_single == beta_id) continue;
          if (beta_id > beta_single && updated_betas.count(beta_id) &&
              updated_betas.count(beta_single)) {
            continue;
          }
          omp_set_lock(&locks[beta_id]);
          beta_id_to_single_ids[beta_id].push_back(beta_single);
          omp_unset_lock(&locks[beta_id]);
          omp_set_lock(&locks[beta_single]);
          beta_id_to_single_ids[beta_single].push_back(beta_id);
          omp_unset_lock(&locks[beta_single]);
        }
      }
      half_det.set(dn_elecs[j]);
    }
  }

  for (auto& lock : locks) omp_destroy_lock(&lock);

  // Sort updated alpha/beta singles and keep uniques.
  unsigned long long singles_cnt = 0;
#pragma omp parallel for schedule(static, 1) reduction(+ : singles_cnt)
  for (size_t alpha_id = 0; alpha_id < n_unique_alphas; alpha_id++) {
    std::sort(alpha_id_to_single_ids[alpha_id].begin(), alpha_id_to_single_ids[alpha_id].end());
    singles_cnt += alpha_id_to_single_ids[alpha_id].size();
  }
#pragma omp parallel for schedule(static, 1) reduction(+ : singles_cnt)
  for (size_t beta_id = 0; beta_id < n_unique_betas; beta_id++) {
    std::sort(beta_id_to_single_ids[beta_id].begin(), beta_id_to_single_ids[beta_id].end());
    singles_cnt += beta_id_to_single_ids[beta_id].size();
  }

  if (is_master) {
    printf(
        "Outer size of a/b singles: %'zu / %'zu\n",
        alpha_id_to_single_ids.size(),
        beta_id_to_single_ids.size());
    printf("Full size of absingles: %'llu\n", singles_cnt);
  }
}

template <class S>
void Hamiltonian<S>::update_matrix(const S& system) {
  const size_t proc_id = Parallel::get_proc_id();
  const size_t n_procs = Parallel::get_n_procs();

#pragma omp parallel for schedule(static, 1)
  for (size_t det_id = proc_id; det_id < n_dets; det_id += n_procs) {
    SparseVector<double>& row = matrix.get_row(det_id);
    const auto& det = hps::parse_from_string<Det>(system.get_det(det_id));
    Det connected_det;
    const bool is_new_det = det_id >= n_dets_prev;
    if (is_new_det) {
      const double H = system.get_hamiltonian_elem(det, det);
      row.append(det_id, H);
    }
    const size_t start_id = is_new_det ? det_id + 1 : n_dets_prev;

    // Single or double alpha excitations.
    const auto& beta = hps::serialize_to_string(det.dn);
    const size_t beta_id = beta_to_id[beta];
    const auto& alpha_dets = beta_id_to_det_ids[beta_id];
    for (auto it = alpha_dets.begin(); it != alpha_dets.end(); it++) {
      const size_t alpha_det_id = *it;
      if (alpha_det_id < start_id) continue;
      hps::parse_from_string(connected_det, system.get_det(alpha_det_id));
      const double H = system.get_hamiltonian_elem(det, connected_det);
      if (std::abs(H) < Util::EPS) continue;
      row.append(alpha_det_id, H);
    }

    // Single or double beta excitations.
    const auto& alpha = hps::serialize_to_string(det.up);
    const size_t alpha_id = alpha_to_id[alpha];
    const auto& beta_dets = alpha_id_to_det_ids[alpha_id];
    for (auto it = beta_dets.begin(); it != beta_dets.end(); it++) {
      const size_t beta_det_id = *it;
      if (beta_det_id < start_id) continue;
      hps::parse_from_string(connected_det, system.get_det(beta_det_id));
      const double H = system.get_hamiltonian_elem(det, connected_det);
      if (std::abs(H) < Util::EPS) continue;
      row.append(beta_det_id, H);
    }

    // Mixed double excitation.
    const auto& alpha_singles = alpha_id_to_single_ids[alpha_id];
    const auto& beta_singles = beta_id_to_single_ids[beta_id];
    for (const auto alpha_single : alpha_singles) {
      const auto& related_beta_ids = alpha_id_to_beta_ids[alpha_single];
      const auto& related_det_ids = alpha_id_to_det_ids[alpha_single];
      const size_t n_related_dets = related_beta_ids.size();
      size_t ptr = 0;
      for (auto it = beta_singles.begin(); it != beta_singles.end(); it++) {
        const size_t beta_single = *it;
        while (ptr < n_related_dets && related_beta_ids[ptr] < beta_single) {
          ptr++;
        }
        if (ptr == n_related_dets) break;

        if (related_beta_ids[ptr] == beta_single) {
          const size_t related_det_id = related_det_ids[ptr];
          ptr++;
          if (related_det_id < start_id) continue;
          hps::parse_from_string(connected_det, system.get_det(related_det_id));
          const double H = system.get_hamiltonian_elem(det, connected_det);
          if (std::abs(H) < Util::EPS) continue;
          row.append(related_det_id, H);
        }
      }
    }
  }
}

template <class S>
void Hamiltonian<S>::sort_by_first(std::vector<size_t>& vec1, std::vector<size_t>& vec2) {
  std::vector<std::pair<size_t, size_t>> vec;
  const size_t n_vec = vec1.size();
  for (size_t i = 0; i < n_vec; i++) {
    vec.push_back(std::make_pair(vec1[i], vec2[i]));
  }
  std::sort(
      vec.begin(),
      vec.end(),
      [&](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
        return a.first < b.first;
      });
  for (size_t i = 0; i < n_vec; i++) {
    vec1[i] = vec[i].first;
    vec2[i] = vec[i].second;
  }
}

