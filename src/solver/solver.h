#pragma once

#include <fgpl/src/broadcast.h>
#include <fgpl/src/dist_hash_map.h>
#include <fgpl/src/dist_hash_set.h>
#include <fgpl/src/dist_range.h>
#include <fgpl/src/hash_set.h>
#include <fgpl/src/reducer.h>
#include <hps/src/hps.h>
#include <omp_hash_map/src/omp_hash_map.h>
#include <omp_hash_map/src/omp_hash_set.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <unordered_set>

#include "../config.h"
#include "../det/det.h"
#include "../math_vector.h"
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "davidson.h"
#include "green.h"
#include "hamiltonian.h"
#include "hc_server.h"
#include "uncert_result.h"

template <class S>
class Solver {
 public:
  void run();

 private:
  S system;

  Hamiltonian<S> hamiltonian;

  std::vector<double> eps_tried_prev;

  fgpl::HashSet<Det, DetHasher> var_dets;

  size_t pt_mem_avail;

  size_t var_iteration_global;

  double eps_var_min;

  double eps_pt;

  double eps_pt_dtm;

  double eps_pt_psto;

  double target_error;

  size_t bytes_per_det;

  void run_all_variations();

  void run_variation(const double eps_var, const bool until_converged = true);

  void run_all_perturbations();

  void run_perturbation(const double eps_var);

  double get_energy_pt_dtm(const double eps_var);

  UncertResult get_energy_pt_psto(const double eps_var, const double energy_pt_dtm);

  UncertResult get_energy_pt_sto(const double eps_var, const UncertResult& get_energy_pt_sto);

  bool load_variation_result(const std::string& filename);

  void save_variation_result(const std::string& filename);

  void save_pair_contrib(const double eps_var);

  void print_dets_info() const;

  std::string get_wf_filename(const double eps_var) const;

  template <class C>
  std::array<double, 2> mapreduce_sum(
      const fgpl::DistHashMap<Det, C, DetHasher>& map,
      const std::function<double(const Det& det, const C& hc_sum)>& mapper) const;
};

template <class S>
void Solver<S>::run() {
  Timer::start("setup");
  std::setlocale(LC_ALL, "en_US.UTF-8");
  system.setup();
  target_error = Config::get<double>("target_error", 5.0e-5);
  Result::put("energy_hf", system.energy_hf);
  Timer::end();

  std::vector<std::vector<size_t>> connections;

  if (!Config::get<bool>("skip_var", false)) {
    Timer::start("variation");
    run_all_variations();

    if (Config::get<bool>("2rdm", false) || Config::get<bool>("get_2rdm_csv", false)) {
      connections = hamiltonian.matrix.get_connections();
    }

    hamiltonian.clear();
    Timer::end();
  }

  Timer::start("post variation");

  if (Config::get<bool>("hc_server_mode", false)) {
    if (system.time_sym) throw std::invalid_argument("time sym hc server not implemented");
    const auto& wf_filename = get_wf_filename(eps_var_min);
    if (!load_variation_result(wf_filename)) throw std::runtime_error("failed to load wf");
    hamiltonian.update(system);
    HcServer<S> server(system, hamiltonian);
    server.run();
    return;
  }

  if (Config::get<bool>("get_green", false)) {
    if (system.time_sym) throw std::invalid_argument("time sym green not implemented");
    Timer::start("green");
    Green<S> green(system, hamiltonian);
    green.run();
    Timer::end();
  }

  system.post_variation(connections);
  connections.clear();
  connections.shrink_to_fit();
  hamiltonian.clear();
  eps_tried_prev.clear();
  var_dets.clear_and_shrink();

  Timer::end();

  if (Config::get<bool>("var_only", false)) return;

  Timer::start("perturbation");
  run_all_perturbations();
  system.post_perturbation();
  Timer::end();
}

template <class S>
void Solver<S>::run_all_variations() {
  if (Parallel::is_master()) {
    printf("Final iteration 0 HF ndets= 1 energy= %.8f\n", system.energy_hf);
  }
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  const auto& eps_vars_schedule = Config::get<std::vector<double>>("eps_vars_schedule");
  double eps_var_prev = Util::INF;
  for (const auto& det : system.dets) var_dets.set(det);
  auto it_schedule = eps_vars_schedule.begin();
  var_iteration_global = 0;
  eps_var_min = eps_vars.back();
  const bool get_pair_contrib = Config::get<bool>("get_pair_contrib", false);
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.2e", eps_var));
    const auto& filename = get_wf_filename(eps_var);
    if (Config::get<bool>("force_var", false) || !load_variation_result(filename)) {
      // Perform extra scheduled eps.
      while (it_schedule != eps_vars_schedule.end() && *it_schedule >= eps_var_prev) it_schedule++;
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var) {
        const double eps_var_extra = *it_schedule;
        Timer::start(Util::str_printf("extra %#.2e", eps_var_extra));
        run_variation(eps_var_extra, false);
        Timer::end();
        it_schedule++;
      }

      Timer::start("main");
      run_variation(eps_var);
      Result::put<double>(Util::str_printf("energy_var/%#.2e", eps_var), system.energy_var);
      Timer::end();
      save_variation_result(filename);
    } else {
      eps_tried_prev.clear();
      var_dets.clear();
      for (const auto& det : system.dets) var_dets.set(det);
      //      hamiltonian.clear();
      Result::put<double>(Util::str_printf("energy_var/%#.2e", eps_var), system.energy_var);
    }

    if (Parallel::is_master() && get_pair_contrib) {
      save_pair_contrib(eps_var);
    }
    eps_var_prev = eps_var;
    Timer::end();
  }

  //  hamiltonian.clear();
  eps_tried_prev.clear();
  eps_tried_prev.shrink_to_fit();
  var_dets.clear_and_shrink();
}

template <class S>
void Solver<S>::run_all_perturbations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  bytes_per_det = N_CHUNKS * 16;
#ifdef INF_ORBS
  bytes_per_det += 128;
#endif
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.2e", eps_var));
    run_perturbation(eps_var);
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var, const bool until_converged) {
  Davidson davidson;
  fgpl::DistHashSet<Det, DetHasher> dist_new_dets;
  size_t n_dets = system.get_n_dets();
  size_t n_dets_new = n_dets;
  double energy_var_prev = 0.0;
  bool converged = false;
  size_t iteration = 0;
  bool dets_converged = false;
  const bool get_pair_contrib = Config::get<bool>("get_pair_contrib", false);
  bool var_sd = Config::get<bool>("var_sd", get_pair_contrib);

  while (!converged) {
    eps_tried_prev.resize(n_dets, Util::INF);
    if (until_converged) Timer::start(Util::str_printf("#%zu", iteration + 1));

    // Random execution and broadcast.
    if (!dets_converged) {
      n_dets_new = n_dets;
      for (size_t j = 0; j < 5; j++) {
        fgpl::DistRange<size_t>(j, n_dets, 5).for_each([&](const size_t i) {
          const auto& det = system.dets[i];
          const double coef = system.coefs[i];
          double eps_min = eps_var / std::abs(coef);
          if (i == 0 && var_sd) eps_min = 0.0;
          if (system.time_sym && det.up != det.dn) eps_min *= Util::SQRT2;
          if (eps_min >= eps_tried_prev[i] * 0.99) return;
          Det connected_det_reg;
          const auto& connected_det_handler = [&](const Det& connected_det, const int n_excite) {
            connected_det_reg = connected_det;
            if (system.time_sym && connected_det.up > connected_det.dn) {
              connected_det_reg.reverse_spin();
            }
            if (var_dets.has(connected_det_reg)) return;
            if (n_excite == 1) {
              const double h_ai = system.get_hamiltonian_elem(det, connected_det, 1);
              if (std::abs(h_ai) < eps_min) return;  // Filter out small single excitation.
            }
            dist_new_dets.async_set(connected_det_reg);
          };
          system.find_connected_dets(det, eps_tried_prev[i], eps_min, connected_det_handler);
          eps_tried_prev[i] = eps_min;
        });
        dist_new_dets.sync();
        n_dets_new += dist_new_dets.get_n_keys();
        system.dets.reserve(n_dets_new);
        system.coefs.reserve(n_dets_new);
        dist_new_dets.for_each_serial([&](const Det& connected_det, const size_t) {
          var_dets.set(connected_det);
          system.dets.push_back(connected_det);
          system.coefs.push_back(1.0e-16);
        });
        dist_new_dets.clear();
        if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
      }

      if (Parallel::is_master()) {
        printf("\nNumber of dets / new dets: %'zu / %'zu\n", n_dets_new, n_dets_new - n_dets);
      }
      Timer::checkpoint("get next det list");

      hamiltonian.update(system);
    }

    const double davidson_target_error = until_converged ? target_error / 5000 : target_error / 50;
    davidson.diagonalize(
        hamiltonian.matrix, system.coefs, davidson_target_error, Parallel::is_master());
    const double energy_var_new = davidson.get_lowest_eigenvalue();
    system.coefs = davidson.get_lowest_eigenvector();
    Timer::checkpoint("diagonalize sparse hamiltonian");
    var_iteration_global++;
    if (Parallel::is_master()) {
      printf("Iteration %zu ", var_iteration_global);
      printf("eps1= %#.2e ndets= %'zu energy= %.8f\n", eps_var, n_dets_new, energy_var_new);
    }
    if (std::abs(energy_var_new - energy_var_prev) < target_error * 0.001) {
      converged = true;
    }
    if (n_dets_new < n_dets * 1.001) {
      dets_converged = true;
      if (davidson.converged) converged = true;
    }
    n_dets = n_dets_new;
    energy_var_prev = energy_var_new;
    if (!until_converged) break;
    Timer::end();
    iteration++;
  }
  system.energy_var = energy_var_prev;
  if (Parallel::is_master() && until_converged) {
    printf("Final iteration %zu ", var_iteration_global);
    printf("eps1= %#.2e ndets= %'zu energy= %.8f\n", eps_var, n_dets, system.energy_var);
    print_dets_info();
  }
}

template <class S>
void Solver<S>::run_perturbation(const double eps_var) {
  double default_eps_pt_dtm = 2.0e-6;
  double default_eps_pt_psto = 1.0e-7;
  double default_eps_pt = eps_var * 1.0e-6;
  if (system.type == SystemType::HEG) {
    default_eps_pt_psto = default_eps_pt_dtm;
    default_eps_pt = eps_var * 1.0e-20;
  }
  eps_pt_dtm = Config::get<double>("eps_pt_dtm", default_eps_pt_dtm);
  eps_pt_psto = Config::get<double>("eps_pt_psto", default_eps_pt_psto);
  eps_pt = Config::get<double>("eps_pt", default_eps_pt);
  if (eps_pt_psto < eps_pt) eps_pt_psto = eps_pt;
  if (eps_pt_dtm < eps_pt_psto) eps_pt_dtm = eps_pt_psto;
  // If result already exists, return.
  const auto& value_entry = Util::str_printf("energy_total/%#.2e/%#.2e/value", eps_var, eps_pt);
  const auto& uncert_entry = Util::str_printf("energy_total/%#.2e/%#.2e/uncert", eps_var, eps_pt);
  UncertResult res(Result::get<double>(value_entry, 0.0));
  if (res.value != 0.0) {
    if (Parallel::is_master()) {
      res.uncert = Result::get<double>(uncert_entry, 0.0);
      printf("Total energy: %s (loaded from result file)\n", res.to_string().c_str());
    }
    if (!Config::get<bool>("force_pt", false)) return;
  }

  // Load var wf.
  const auto& var_filename = get_wf_filename(eps_var);
  if (!load_variation_result(var_filename)) {
    throw new std::runtime_error("cannot load variation results");
  }
  system.update_diag_helper();
  if (system.time_sym) system.unpack_time_sym();

  // Perform multi stage PT.
  system.dets.shrink_to_fit();
  system.coefs.shrink_to_fit();
  var_dets.clear_and_shrink();
  var_dets.reserve(system.get_n_dets());
  for (const auto& det : system.dets) var_dets.set(det);
  size_t mem_total = Config::get<double>("mem_total", Util::get_mem_total());
#ifdef INF_ORBS
  mem_total *= 0.8;
#endif
  const size_t mem_var = system.get_n_dets() * (bytes_per_det * 3 + 8);
  const double mem_left = mem_total * 0.7 - mem_var - system.helper_size;
  assert(mem_left > 0);
  pt_mem_avail = mem_left;
  const size_t n_procs = Parallel::get_n_procs();
  if (n_procs >= 2) {
    pt_mem_avail = static_cast<size_t>(pt_mem_avail * 0.7 * n_procs);
  }
  if (Parallel::is_master()) {
    printf("Memory total: %.1fGB\n", mem_total * 1.0e-9);
    printf("Helper size: %.1fGB\n", system.helper_size * 1.0e-9);
    printf("Bytes per det: %zu\n", bytes_per_det);
    printf("Memory var: %.1fGB\n", mem_var * 1.0e-9);
    printf("Memory PT limit: %.1fGB\n", pt_mem_avail * 1.0e-9);
  }
  const double energy_pt_dtm = get_energy_pt_dtm(eps_var);
  const UncertResult energy_pt_psto = get_energy_pt_psto(eps_var, energy_pt_dtm);
  const UncertResult energy_pt = get_energy_pt_sto(eps_var, energy_pt_psto);
  if (Parallel::is_master()) {
    printf("Total energy: %s Ha\n", energy_pt.to_string().c_str());
  }
  Result::put(value_entry, energy_pt.value);
  Result::put(uncert_entry, energy_pt.uncert);
}

template <class S>
double Solver<S>::get_energy_pt_dtm(const double eps_var) {
  if (eps_pt_dtm >= 1.0) return system.energy_var;
  Timer::start(Util::str_printf("dtm %#.2e", eps_pt_dtm));
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_dtm", 0);
  fgpl::DistHashMap<Det, MathVector<double, 1>, DetHasher> hc_sums;
  size_t bytes_per_entry = bytes_per_det + 8;
  const DetHasher det_hasher;

  // Estimate best n batches.
  if (n_batches == 0) {
    fgpl::DistRange<size_t>(50, n_var_dets, 100).for_each([&](const size_t i) {
      const Det& det = system.dets[i];
      const double coef = system.coefs[i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;  // use 1st of 16 batches.
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_dtm) return;  // Filter out small single excitation.
        }
        MathVector<double, 1> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      system.find_connected_dets(det, Util::INF, eps_pt_dtm / std::abs(coef), pt_det_handler);
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    n_batches =
        static_cast<size_t>(ceil(2.0 * 128 * 100 * n_pt_dets * bytes_per_entry / pt_mem_avail));
    if (n_batches == 0) n_batches = 1;
    if (Parallel::is_master()) {
      printf("Number of dtm batches: %zu\n", n_batches);
    }
    Timer::checkpoint("determine number of dtm batches");
    hc_sums.clear();
  }

  double energy_sum = 0.0;
  double energy_sq_sum = 0.0;
  size_t n_pt_dets_sum = 0;
  UncertResult energy_pt_dtm;

  for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
    Timer::start(Util::str_printf("#%zu/%zu", batch_id + 1, n_batches));

    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_var_dets, 5).for_each([&](const size_t i) {
        const Det& det = system.dets[i];
        const double coef = system.coefs[i];
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_dtm) return;  // Filter out small single excitation.
          const MathVector<double, 1> contrib(hc);
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 1>>::sum);
        };
        system.find_connected_dets(det, Util::INF, eps_pt_dtm / std::abs(coef), pt_det_handler);
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 1>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) {
      printf("\nNumber of dtm pt dets: %'zu\n", n_pt_dets);
    }
    n_pt_dets_sum += n_pt_dets;
    Timer::checkpoint("create hc sums");

    const auto& energy_pt_dtm_batch = mapreduce_sum<MathVector<double, 1>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 1>& hc_sum) {
          const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double contrib = hc_sum[0] * hc_sum[0] / (system.energy_var - H_aa);
          return contrib;
        });
    energy_sum += energy_pt_dtm_batch[0];
    energy_sq_sum += energy_pt_dtm_batch[1];
    energy_pt_dtm.value = energy_sum / (batch_id + 1) * n_batches;
    if (batch_id == n_batches - 1) {
      energy_pt_dtm.uncert = 0.0;
    } else {
      const double energy_avg = energy_sum / n_pt_dets_sum;
      const double sample_stdev = sqrt(energy_sq_sum / n_pt_dets_sum - energy_avg * energy_avg);
      energy_pt_dtm.uncert =
          sample_stdev * sqrt(n_pt_dets_sum) / (batch_id + 1) * (n_batches - batch_id - 1);
    }

    if (Parallel::is_master()) {
      printf("PT dtm batch correction: " ENERGY_FORMAT "\n", energy_pt_dtm_batch[0]);
      printf("PT dtm correction (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", energy_pt_dtm.to_string().c_str());
      printf("PT dtm total energy (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", (energy_pt_dtm + system.energy_var).to_string().c_str());
      printf("Correlation energy (eps1= %.2e):", eps_var);
      printf(
          " %s Ha\n", (energy_pt_dtm + system.energy_var - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();  // batch
  }

  hc_sums.clear_and_shrink();
  Timer::end();  // dtm
  return energy_pt_dtm.value + system.energy_var;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_psto(const double eps_var, const double energy_pt_dtm) {
  if (eps_pt_psto >= eps_pt_dtm) return UncertResult(energy_pt_dtm, 0.0);

  Timer::start(Util::str_printf("psto %#.2e", eps_pt_psto));
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_psto", 0);
  fgpl::DistHashMap<Det, MathVector<double, 2>, DetHasher> hc_sums;
  const size_t bytes_per_entry = bytes_per_det + 16;
  const DetHasher det_hasher;

  // Estimate best n batches.
  if (n_batches == 0) {
    fgpl::DistRange<size_t>(50, n_var_dets, 100).for_each([&](const size_t i) {
      const Det& det = system.dets[i];
      const double coef = system.coefs[i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;  // use 1st of 16 batches.
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_psto) return;  // Filter out small single excitation.
        }
        MathVector<double, 2> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      system.find_connected_dets(det, Util::INF, eps_pt_psto / std::abs(coef), pt_det_handler);
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    const double mem_usage = Config::get<double>("pt_psto_mem_usage", 1.0);
    n_batches = static_cast<size_t>(
        ceil(2.0 * 128 * 100 * n_pt_dets * bytes_per_entry / (pt_mem_avail * mem_usage)));
    if (n_batches < 16) n_batches = 16;
    if (Parallel::is_master()) {
      printf("Number of psto batches: %zu\n", n_batches);
    }
    Timer::checkpoint("determine number of psto batches");
    hc_sums.clear();
  }

  double energy_sum = 0.0;
  double energy_sq_sum = 0.0;
  size_t n_pt_dets_sum = 0;
  UncertResult energy_pt_psto;

  for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
    Timer::start(Util::str_printf("#%zu/%zu", batch_id + 1, n_batches));

    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_var_dets, 5).for_each([&](const size_t i) {
        const Det& det = system.dets[i];
        const double coef = system.coefs[i];
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt_psto) return;  // Filter out small single excitation.
          MathVector<double, 2> contrib;
          contrib[0] = hc;
          if (std::abs(hc) >= eps_pt_dtm) contrib[1] = hc;
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 2>>::sum);
        };
        system.find_connected_dets(det, Util::INF, eps_pt_psto / std::abs(coef), pt_det_handler);
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 2>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) {
      printf("\nNumber of psto pt dets: %'zu\n", n_pt_dets);
    }
    n_pt_dets_sum += n_pt_dets;
    Timer::checkpoint("create hc sums");

    const auto& energy_pt_psto_batch = mapreduce_sum<MathVector<double, 2>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 2>& hc_sum) {
          const double hc_sum_sq_diff = hc_sum[0] * hc_sum[0] - hc_sum[1] * hc_sum[1];
          const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double contrib = hc_sum_sq_diff / (system.energy_var - H_aa);
          return contrib;
        });
    energy_sum += energy_pt_psto_batch[0];
    energy_sq_sum += energy_pt_psto_batch[1];
    energy_pt_psto.value = energy_sum / (batch_id + 1) * n_batches;
    if (batch_id == n_batches - 1) {
      energy_pt_psto.uncert = 0.0;
    } else {
      const double energy_avg = energy_sum / n_pt_dets_sum;
      const double sample_stdev = sqrt(energy_sq_sum / n_pt_dets_sum - energy_avg * energy_avg);
      const double mean_stdev = sample_stdev / sqrt(n_pt_dets_sum);
      energy_pt_psto.uncert =
          mean_stdev * n_pt_dets_sum / (batch_id + 1) * (n_batches - batch_id - 1);
      // energy_pt_psto.uncert = sample_stdev * sqrt(n_pt_dets_sum) / (batch_id + 1) * n_batches;
    }

    if (Parallel::is_master()) {
      printf("PT psto batch correction: " ENERGY_FORMAT "\n", energy_pt_psto_batch[0]);
      printf("PT psto correction (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", energy_pt_psto.to_string().c_str());
      printf("PT psto total energy (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", (energy_pt_psto + energy_pt_dtm).to_string().c_str());
      printf("Correlation energy (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", (energy_pt_psto + energy_pt_dtm - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();  // batch

    if (energy_pt_psto.uncert <= target_error * 0.7) break;
    if (eps_pt_psto <= eps_pt && energy_pt_psto.uncert <= target_error) break;
  }

  Timer::end();  // psto
  return energy_pt_psto + energy_pt_dtm;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_sto(
    const double eps_var, const UncertResult& energy_pt_psto) {
  if (eps_pt >= eps_pt_psto) return energy_pt_psto;

  const size_t max_pt_iterations = Config::get<size_t>("max_pt_iterations", 100);
  fgpl::DistHashMap<Det, MathVector<double, 3>, DetHasher> hc_sums;
  const size_t bytes_per_entry = bytes_per_det + 24;
  const size_t n_var_dets = system.get_n_dets();
  size_t n_batches = Config::get<size_t>("n_batches_pt_sto", 0);
  if (n_batches == 0) n_batches = 64;
  size_t n_samples = Config::get<size_t>("n_samples_pt_sto", 0);
  std::vector<double> probs(n_var_dets);
  std::vector<double> cum_probs(n_var_dets);  // For sampling.
  std::unordered_map<size_t, unsigned> sample_dets;
  std::vector<size_t> sample_dets_list;
  size_t iteration = 0;
  const DetHasher det_hasher;

  UncertResult energy_pt_sto;
  std::vector<double> energy_pt_sto_loops;

  // Contruct probs.
  double sum_weights = 0.0;
  for (size_t i = 0; i < n_var_dets; i++) {
    sum_weights += std::abs(system.coefs[i]);
  }
  for (size_t i = 0; i < n_var_dets; i++) {
    probs[i] = std::abs(system.coefs[i]) / sum_weights;
    cum_probs[i] = probs[i];
    if (i > 0) cum_probs[i] += cum_probs[i - 1];
  }

  const unsigned random_seed = Config::get<unsigned>("random_seed", time(nullptr));
  srand(random_seed);
  Timer::start(Util::str_printf("sto %#.2e", eps_pt));

  // Estimate best n sample.
  if (n_samples == 0) {
    for (size_t i = 0; i < 1000; i++) {
      const double rand_01 = (static_cast<double>(rand()) / (RAND_MAX));
      const int sample_det_id =
          std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
      if (sample_dets.count(sample_det_id) == 0) sample_dets_list.push_back(sample_det_id);
      sample_dets[sample_det_id]++;
    }
    fgpl::broadcast(sample_dets);
    fgpl::broadcast(sample_dets_list);
    size_t n_unique_samples = sample_dets_list.size();
    fgpl::DistRange<size_t>(0, n_unique_samples).for_each([&](const size_t sample_id) {
      const size_t i = sample_dets_list[sample_id];
      const Det& det = system.dets[i];
      const double coef = system.coefs[i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if ((batch_hash & 127) != 0) return;
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt) return;  // Filter out small single excitation.
        }
        MathVector<double, 3> contrib;
        hc_sums.async_set(det_a, contrib);
      };
      system.find_connected_dets(det, Util::INF, eps_pt / std::abs(coef), pt_det_handler);
    });
    hc_sums.sync();
    const size_t n_pt_dets = hc_sums.get_n_keys();
    hc_sums.clear();
    const size_t n_pt_dets_batch = n_pt_dets * 128 / n_batches;
    double default_mem_usage = 0.4;
    if (system.type == SystemType::HEG) default_mem_usage = 1.0;
    const double mem_usage = Config::get<double>("pt_sto_mem_usage", default_mem_usage);
    size_t n_unique_target =
        pt_mem_avail * mem_usage * n_unique_samples / bytes_per_entry / 5.0 / n_pt_dets_batch;
    const size_t max_unique_targets = n_var_dets / 8 + 1;
    if (n_unique_target >= max_unique_targets) n_unique_target = max_unique_targets;
    sample_dets.clear();
    sample_dets_list.clear();
    n_samples = 0;
    n_unique_samples = 0;
    while (n_unique_samples < n_unique_target) {
      const double rand_01 = (static_cast<double>(rand()) / (RAND_MAX));
      const int sample_det_id =
          std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
      if (sample_dets.count(sample_det_id) == 0) {
        n_unique_samples++;
      }
      n_samples++;
      sample_dets[sample_det_id]++;
    }
    sample_dets.clear();
    fgpl::broadcast(n_samples);
    if (Parallel::is_master()) {
      printf("Number of samples chosen: %'zu\n", n_samples);
    }
    Timer::checkpoint("determine n samples");
  }

  while (iteration < max_pt_iterations) {
    Timer::start(Util::str_printf("#%zu", iteration + 1));

    // Generate random sample
    for (size_t i = 0; i < n_samples; i++) {
      const double rand_01 = (static_cast<double>(rand()) / (RAND_MAX));
      const int sample_det_id =
          std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
      if (sample_dets.count(sample_det_id) == 0) sample_dets_list.push_back(sample_det_id);
      sample_dets[sample_det_id]++;
    }
    fgpl::broadcast(sample_dets);
    fgpl::broadcast(sample_dets_list);
    if (Parallel::is_master()) {
      printf(
          "Number of unique variational determinants in sample: %'zu\n", sample_dets_list.size());
    }

    // Select random batch.
    size_t batch_id = rand() % n_batches;
    fgpl::broadcast(batch_id);
    const size_t n_unique_samples = sample_dets_list.size();
    if (Parallel::is_master()) printf("Batch id: %zu / %zu\n", batch_id, n_batches);

    for (size_t j = 0; j < 5; j++) {
      fgpl::DistRange<size_t>(j, n_unique_samples, 5).for_each([&](const size_t sample_id) {
        const size_t i = sample_dets_list[sample_id];
        const double count = static_cast<double>(sample_dets[i]);
        const Det& det = system.dets[i];
        const double coef = system.coefs[i];
        const double prob = probs[i];
        const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
          const size_t det_a_hash = det_hasher(det_a);
          const size_t batch_hash = Util::rehash(det_a_hash);
          if (batch_hash % n_batches != batch_id) return;
          if (var_dets.has(det_a)) return;
          const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
          const double hc = h_ai * coef;
          if (std::abs(hc) < eps_pt) return;  // Filter out small single excitation.
          const double factor = static_cast<double>(n_batches) / (n_samples * (n_samples - 1));

          MathVector<double, 3> contrib;
          contrib[0] = count * hc / prob * sqrt(factor);
          if (std::abs(hc) < eps_pt_psto) {
            contrib[2] = (n_samples - 1 - count / prob) * hc * hc * factor * count / prob;
          } else {
            contrib[1] = contrib[0];
          }
          hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 3>>::sum);
        };
        system.find_connected_dets(det, Util::INF, eps_pt / std::abs(coef), pt_det_handler);
      });
      hc_sums.sync(fgpl::Reducer<MathVector<double, 3>>::sum);
      if (Parallel::is_master()) printf("%zu%% ", (j + 1) * 20);
    }
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) printf("\nNumber of sto pt dets: %'zu\n", n_pt_dets);
    sample_dets.clear();
    sample_dets_list.clear();
    Timer::checkpoint("create hc sums");

    const double energy_pt_sto_loop = mapreduce_sum<MathVector<double, 3>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 3>& hc_sum) {
          const double h_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double factor = 1.0 / (system.energy_var - h_aa);
          return (hc_sum[0] * hc_sum[0] - hc_sum[1] * hc_sum[1] + hc_sum[2]) * factor;
        })[0];

    energy_pt_sto_loops.push_back(energy_pt_sto_loop);
    energy_pt_sto.value = Util::avg(energy_pt_sto_loops);
    energy_pt_sto.uncert = Util::stdev(energy_pt_sto_loops) / sqrt(iteration + 1.0);
    if (Parallel::is_master()) {
      printf("PT sto loop correction: " ENERGY_FORMAT "\n", energy_pt_sto_loop);
      printf("PT sto correction (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", energy_pt_sto.to_string().c_str());
      printf("PT sto total energy (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", (energy_pt_sto + energy_pt_psto).to_string().c_str());
      printf("Correlation energy (eps1= %.2e):", eps_var);
      printf(" %s Ha\n", (energy_pt_sto + energy_pt_psto - system.energy_hf).to_string().c_str());
    }

    hc_sums.clear();
    Timer::end();
    iteration++;
    if (iteration >= 6 && (energy_pt_sto + energy_pt_psto).uncert <= target_error * 0.5) {
      break;
    }
    if (iteration >= 10 && (energy_pt_sto + energy_pt_psto).uncert <= target_error) {
      break;
    }
  }

  hc_sums.clear_and_shrink();
  Timer::end();
  return energy_pt_sto + energy_pt_psto;
}

template <class S>
template <class C>
std::array<double, 2> Solver<S>::mapreduce_sum(
    const fgpl::DistHashMap<Det, C, DetHasher>& map,
    const std::function<double(const Det& det, const C& hc_sum)>& mapper) const {
  const int n_threads = omp_get_max_threads();
  std::vector<double> res_sq_thread(n_threads, 0.0);
  std::vector<double> res_thread(n_threads, 0.0);
  map.for_each([&](const Det& key, const size_t, const C& value) {
    const int thread_id = omp_get_thread_num();
    const double mapped = mapper(key, value);
    res_thread[thread_id] += mapped;
    res_sq_thread[thread_id] += mapped * mapped;
  });
  std::array<double, 2> res_local = {0.0, 0.0};
  std::array<double, 2> res;
  for (int i = 0; i < n_threads; i++) {
    res_local[0] += res_thread[i];
    res_local[1] += res_sq_thread[i];
  }
  MPI_Allreduce(&res_local, &res, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return res;
}

template <class S>
bool Solver<S>::load_variation_result(const std::string& filename) {
  std::string serialized;
  const int TRUNK_SIZE = 1 << 20;
  char buffer[TRUNK_SIZE];
  MPI_File file;
  int error;
  error = MPI_File_open(
      MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY | MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  if (error) return false;
  MPI_Offset size;
  MPI_File_get_size(file, &size);
  MPI_Status status;
  while (size > TRUNK_SIZE) {
    MPI_File_read_all(file, buffer, TRUNK_SIZE, MPI_CHAR, &status);
    serialized.append(buffer, TRUNK_SIZE);
    size -= TRUNK_SIZE;
  }
  MPI_File_read_all(file, buffer, size, MPI_CHAR, &status);
  serialized.append(buffer, size);
  MPI_File_close(&file);
  hps::from_string(serialized, system);
  if (Parallel::is_master()) {
    printf("Loaded %'zu dets from: %s\n", system.get_n_dets(), filename.c_str());
    print_dets_info();
    printf("HF energy: " ENERGY_FORMAT "\n", system.energy_hf);
    printf("Variational energy: " ENERGY_FORMAT "\n", system.energy_var);
  }
  return true;
}

template <class S>
void Solver<S>::save_variation_result(const std::string& filename) {
  if (Parallel::is_master()) {
    std::ofstream file(filename, std::ofstream::binary);
    hps::to_stream(system, file);
    printf("Variational results saved to: %s\n", filename.c_str());
  }
}

template <class S>
void Solver<S>::save_pair_contrib(const double eps_var) {
  const auto& det_hf = system.dets[0];
  // const size_t n_elecs = system.n_elecs;
  const size_t n_up = system.n_up;
  // const size_t n_dn = system.n_dn;
  if (det_hf.up != det_hf.dn) {
    throw std::invalid_argument("non sym det_hf not implemented");
  }
  std::vector<std::vector<double>> contribs(n_up);
  std::string contrib_filename = Util::str_printf("pair_contrib_%#.2e.csv", eps_var);
  const auto& contrib_entry = Util::str_printf("pair_contrib/%#.2e", eps_var);
  Result::put<std::string>(contrib_entry, contrib_filename);
  std::ofstream contrib_file(contrib_filename);
  contrib_file << "i,j,pair_contrib" << std::endl;
  for (size_t i = 0; i < n_up; i++) {
    contribs[i].assign(n_up, 0.0);
  }
  const double c0 = system.coefs[0];
  for (size_t det_id = 1; det_id < system.dets.size(); det_id++) {
    const auto& det = system.dets[det_id];
    const double coef = system.coefs[det_id];
    const auto& diff_up = det_hf.up.diff(det.up);
    const auto& diff_dn = det_hf.dn.diff(det.dn);
    const unsigned n_excite = diff_up.n_diffs + diff_dn.n_diffs;
    if (n_excite > 2) continue;
    size_t i = 0;
    size_t j = 0;
    const auto& H = system.get_hamiltonian_elem(det_hf, det, -1);
    if (diff_up.n_diffs == 2) {
      i = diff_up.left_only[0];
      j = diff_up.left_only[1];
    } else if (diff_up.n_diffs == 1) {
      i = diff_up.left_only[0];
      if (diff_dn.n_diffs == 1) {
        j = diff_dn.left_only[0];
        if (j < i) {
          std::swap(i, j);
        }
      } else {
        j = i;
      }
    } else {
      i = diff_dn.left_only[0];
      if (diff_dn.n_diffs == 2) {
        j = diff_dn.left_only[1];
        if (j < i) {
          std::swap(i, j);
        }
      } else {
        j = i;
      }
    }
    if (det.up == det.dn) {
      contribs[i][j] += H * coef / c0;
    } else if (system.time_sym) {
      contribs[i][j] += H * coef / c0 * Util::SQRT2;
    } else {
      contribs[i][j] += H * coef / c0;
    }
  }
  contrib_file.precision(15);
  for (size_t i = 0; i < n_up; i++) {
    for (size_t j = i; j < n_up; j++) {
      if (i != j) {
        contribs[i][j] /= 2;
      }
      contrib_file << i << "," << j << "," << contribs[i][j] << std::endl;
    }
  }
  contrib_file.close();
}

template <class S>
void Solver<S>::print_dets_info() const {
  if (system.time_sym) {
    // Print effective dets for unpacked time sym.
    size_t n_eff_dets = 0;
    for (const auto& det : system.dets) {
      if (det.up == det.dn) {
        n_eff_dets += 1;
      } else if (det.up < det.dn) {
        n_eff_dets += 2;
      } else {
        throw std::runtime_error("wf has unvalid det for time sym");
      }
    }
    printf("Effect dets (without time sym): %'zu\n", n_eff_dets);
  }

  // Print excitations.
  std::unordered_map<unsigned, size_t> excitations;
  std::unordered_map<unsigned, double> weights;
  unsigned highest_excitation = 0;
  const auto& det_hf = system.dets[0];
  for (size_t i = 0; i < system.dets.size(); i++) {
    const auto& det = system.dets[i];
    const double coef = system.coefs[i];
    const unsigned n_excite = det_hf.up.n_diffs(det.up) + det_hf.dn.n_diffs(det.dn);
    if (det.up != det.dn && system.time_sym) {
      excitations[n_excite] += 2;
    } else {
      excitations[n_excite] += 1;
    }
    weights[n_excite] += coef * coef;
    if (highest_excitation < n_excite) highest_excitation = n_excite;
  }
  printf("----------------------------------------\n");
  printf("%-10s%12s%16s\n", "Excite Lv", "# dets", "Sum c^2");
  for (unsigned i = 0; i <= highest_excitation; i++) {
    if (excitations.count(i) == 0) {
      excitations[i] = 0;
      weights[i] = 0.0;
    }
    printf("%-10u%12zu%16.8f\n", i, excitations[i], weights[i]);
  }

  // Print orb occupations.
  std::vector<double> orb_occupations(system.n_orbs, 0.0);
  for (size_t i = 0; i < system.dets.size(); i++) {
    const auto& det = system.dets[i];
    const double coef = system.coefs[i];
    for (unsigned j = 0; j < system.n_orbs; j++) {
      if (det.up.has(j)) {
        orb_occupations[j] += coef * coef;
      }
      if (det.dn.has(j)) {
        orb_occupations[j] += coef * coef;
      }
    }
  }
  printf("----------------------------------------\n");
  printf("%-10s%12s%16s\n", "Orbital", "", "Sum c^2");
  for (unsigned j = 0; j < system.n_orbs && j < 50; j++) {
    printf("%-10u%12s%16.8f\n", j, "", orb_occupations[j]);
  }
  double sum_orb_occupation = std::accumulate(orb_occupations.begin(), orb_occupations.end(), 0.0);
  printf("Sum orbitals c^2: %.8f\n", sum_orb_occupation);

  // Print most important dets.
  printf("----------------------------------------\n");
  printf("Most important dets:\n");
  std::vector<size_t> det_order(system.dets.size());
  for (size_t i = 0; i < system.dets.size(); i++) {
    det_order[i] = i;
  }
  std::stable_sort(det_order.begin(), det_order.end(), [&](const size_t a, const size_t b) {
    return std::abs(system.coefs[a]) > std::abs(system.coefs[b]);
  });
  printf("%-10s%12s      %-12s\n", "Excite Lv", "Coef", "Det (Reordered orb)");
  for (size_t i = 0; i < std::min((size_t)20, system.dets.size()); i++) {
    const double coef = system.coefs[det_order[i]];
    const auto& det = system.dets[det_order[i]];
    const auto& occs_up = det.up.get_occupied_orbs();
    const auto& occs_dn = det.dn.get_occupied_orbs();
    const unsigned n_excite = det_hf.up.n_diffs(det.up) + det_hf.dn.n_diffs(det.dn);
    printf("%-10u%12.8f", n_excite, coef);
    printf("      | ");
    for (unsigned j = 0; j < system.n_up; j++) {
      printf("%2u ", occs_up[j]);
    }
    printf("| ");
    for (unsigned j = 0; j < system.n_dn; j++) {
      printf("%2u ", occs_dn[j]);
    }
    printf("|\n");
  }
  printf("----------------------------------------\n");
}

template <class S>
std::string Solver<S>::get_wf_filename(const double eps_var) const {
  return Util::str_printf("wf_eps1_%#.2e.dat", eps_var);
}
