#pragma once

#include <fgpl/src/hash_set.h>
#include <hps/src/hps.h>
#include <omp_hash_map/src/omp_hash_map.h>
#include <omp_hash_map/src/omp_hash_set.h>
#include <cmath>
#include <functional>
#include <unordered_set>
#include "../config.h"
#include "../det/det.h"
#include "../parallel.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "davidson.h"
#include "hamiltonian.h"
#include "uncert_result.h"

template <class S>
class Solver {
 public:
  void run();

 private:
  S system;

  Hamiltonian<S> hamiltonian;

  std::vector<double> eps_tried_prev;

  std::unordered_set<Det, DetHasher> var_dets;

  size_t var_iteration_global;

  void run_all_variations();

  void run_variation(const double eps_var, const bool until_converged = true);

  void run_all_perturbations();

  void run_perturbation(const double eps_var);

  double get_energy_pt_pre_dtm();

  UncertResult get_energy_pt_dtm(const double energy_pt_pre_dtm);

  UncertResult get_energy_pt_sto(const UncertResult& get_energy_pt_sto);

  bool load_variation_result(const std::string& filename);

  void save_variation_result(const std::string& filename);
};

template <class S>
void Solver<S>::run() {
  Timer::start("setup");
  std::setlocale(LC_ALL, "en_US.UTF-8");
  system.setup();
  Result::put("energy_hf", system.energy_hf);
  system.dets[0].up.print();
  system.dets[0].dn.print();
  Timer::end();

  Timer::start("variation");
  run_all_variations();
  hamiltonian.clear();
  Timer::end();

  if (Config::get<bool>("var_only", false)) return;

  // Timer::start("perturbation");
  // run_all_perturbations();
  // Timer::end();

  // Result::dump();
}

template <class S>
void Solver<S>::run_all_variations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  const auto& eps_vars_schedule = Config::get<std::vector<double>>("eps_vars_schedule");
  double eps_var_prev = Util::INF;
  for (const auto& det : system.dets) var_dets.insert(det);
  auto it_schedule = eps_vars_schedule.begin();
  var_iteration_global = 0;
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.4g", eps_var));
    const auto& filename = Util::str_printf("var_%#.4g.dat", eps_var);
    if (!load_variation_result(filename)) {
      // Perform extra scheduled eps.
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var_prev) it_schedule++;
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var) {
        const double eps_var_extra = *it_schedule;
        Timer::start(Util::str_printf("extra %#.4g", eps_var_extra));
        run_variation(eps_var_extra, false);
        Timer::end();
        it_schedule++;
      }

      Timer::start("main");
      run_variation(eps_var);
      Result::put<double>(Util::str_printf("energy_var/%#.4g", eps_var), system.energy_var);
      Timer::end();

      save_variation_result(filename);
    } else {
      hamiltonian.clear();
    }
    eps_var_prev = eps_var;
    Timer::end();
  }
}

// template <class S>
// void Solver<S>::run_all_perturbations() {
//   const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
//   for (const double eps_var : eps_vars) {
//     Timer::start(Util::str_printf("eps_var %#.4g", eps_var));
//     run_perturbation(eps_var);
//     Timer::end();
//   }
// }

template <class S>
void Solver<S>::run_variation(const double eps_var, const bool until_converged) {
  Davidson davidson;
  fgpl::DistHashSet<Det> dist_new_dets;
  const auto& connected_det_handler = [&](const Det& connected_det, const int) {
    if (var_dets.count(connected_det) == 1) return;
    dist_new_dets.async_set(connected_det);
  };
  size_t n_dets = system.get_n_dets();
  double energy_var_prev = 0.0;
  bool converged = false;
  size_t iteration = 0;

  while (!converged) {
    eps_tried_prev.resize(n_dets, Util::INF);
    Timer::start(Util::str_printf("#%zu", iteration));

    // Random execution and broadcast.
    fgpl::DistRange<size_t>(0, n_dets).for_each([&](const size_t i) {
      const double coef = system.coefs[i];
      const double eps_min = eps_var / std::abs(coef);
      if (eps_min >= eps_tried_prev[i]) return;
      const auto& det = system.dets[i];
      system.find_connected_dets(det, eps_tried_prev[i], eps_min, connected_det_handler);
      eps_tried_prev[i] = eps_min;
    });

    dist_new_dets.sync();
    dist_new_dets.to_serial().for_each([&](const Det& connected_det, const size_t) {
      var_dets.insert(connected_det);
      system.dets.push_back(connected_det);
      system.coefs.push_back(0.0);
    });
    dist_new_dets.clear();

    // const size_t n_dets_new = system.get_n_dets();
    const size_t n_dets_new = system.coefs.size();
    if (Parallel::is_master()) {
      printf("Number of dets / new dets: %'zu / %'zu\n", n_dets_new, n_dets_new - n_dets);
    }

    hamiltonian.update(system);
    davidson.diagonalize(hamiltonian.matrix, system.coefs, Parallel::is_master());
    const double energy_var_new = davidson.get_lowest_eigenvalue();
    system.coefs = davidson.get_lowest_eigenvector();
    Timer::checkpoint("hamiltonian diagonalized");
    var_iteration_global++;
    if (Parallel::is_master()) {
      printf("Current variational energy: " ENERGY_FORMAT "\n", energy_var_new);
      printf(
          "Summary: Iteration %zu eps1= %#.2e ndets= %zu energy= %.8f\n",
          var_iteration_global,
          eps_var,
          n_dets_new,
          energy_var_new);
    }
    if (std::abs(energy_var_new - energy_var_prev) < 1.0e-6) {
      converged = true;
    }
    n_dets = n_dets_new;
    energy_var_prev = energy_var_new;
    Timer::end();
    if (!until_converged) break;
    iteration++;
  }
  system.energy_var = energy_var_prev;
}

// template <class S>
// void Solver<S>::run_perturbation(const double eps_var) {
//   // If result already exists, return.
//   const double eps_pt = Config::get<double>("eps_pt");
//   const auto& value_entry = Util::str_printf("energy_pt/%#.4g/%#.4g/value", eps_var, eps_pt);
//   const auto& uncert_entry = Util::str_printf("energy_pt/%#.4g/%#.4g/uncert", eps_var,
//   eps_pt); UncertResult res(Result::get<double>(value_entry, 0.0)); if (res.value != 0.0) {
//     if (Parallel::is_master()) {
//       res.uncert = Result::get<double>(uncert_entry, 0.0);
//       printf("PT energy: %s (loaded from result file)\n", res.to_string().c_str());
//     }
//     // return;
//   }

//   // Load var wf.
//   const auto& var_filename = Util::str_printf("var_%#.4g.dat", eps_var);
//   if (!load_variation_result(var_filename)) {
//     throw new std::runtime_error("cannot load variation results");
//   }

//   // Perform multi stage PT.
//   var_det_strs.clear();
//   for (const auto& det_str : system.det_strs) var_det_strs.insert(det_str);
//   const double energy_pt_pre_dtm = get_energy_pt_pre_dtm();
//   const UncertResult energy_pt_dtm = get_energy_pt_dtm(energy_pt_pre_dtm);
//   const UncertResult energy_pt = get_energy_pt_sto(energy_pt_dtm);
//   if (Parallel::is_master()) {
//     printf("PT energy: %s Ha\n", energy_pt.to_string().c_str());
//   }
//   Result::put(value_entry, energy_pt.value);
//   Result::put(uncert_entry, energy_pt.uncert);
// }

// template <class S>
// double Solver<S>::get_energy_pt_pre_dtm() {
//   const double eps_pt_pre_dtm = Config::get<double>("eps_pt_pre_dtm");
//   Timer::start(Util::str_printf("pre dtm %#.4g", eps_pt_pre_dtm));
//   const size_t n_var_dets = system.get_n_dets();
//   omp_hash_map<std::string, double> hc_sums;

//   Timer::start("search");
// #pragma omp parallel for schedule(dynamic, 5)
//   for (size_t i = 0; i < n_var_dets; i++) {
//     const Det& var_det = system.get_det(i);
//     const double coef = system.coefs[i];
//     const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
//       const auto& det_a_code = hps::to_string(det_a);
//       if (var_det_strs.count(det_a_code) == 1) return;
//       const double h_ai = system.get_hamiltonian_elem(var_det, det_a, n_excite);
//       const double hc = h_ai * coef;
//       hc_sums.set(det_a_code, [&](double& value) { value += hc; }, 0.0);
//     };
//     system.find_connected_dets(var_det, Util::INF, eps_pt_pre_dtm / std::abs(coef),
//     pt_det_handler);
//   }
//   if (Parallel::is_master()) {
//     printf("Number of pre dtm pt dets: %'zu\n", hc_sums.get_n_keys());
//   }
//   Timer::end();  // search

//   Timer::start("accumulate");
//   double energy_pt_pre_dtm = 0.0;
//   hc_sums.apply([&](const std::string& det_a_code, const double hc_sum) {
//     const auto& det_a = hps::from_string<Det>(det_a_code);
//     const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
//     const double contribution = hc_sum * hc_sum / (system.energy_var - H_aa);
// #pragma omp atomic
//     energy_pt_pre_dtm += contribution;
//   });
//   if (Parallel::is_master()) {
//     printf("PT pre dtm correction: " ENERGY_FORMAT "\n", energy_pt_pre_dtm);
//     printf("PT pre dtm energy: " ENERGY_FORMAT "\n", energy_pt_pre_dtm + system.energy_var);
//   }
//   Timer::end();  // accumulate

//   Timer::end();  // pre dtm
//   return energy_pt_pre_dtm + system.energy_var;
// }

// template <class S>
// UncertResult Solver<S>::get_energy_pt_dtm(const double energy_pt_pre_dtm) {
//   const double eps_pt_dtm = Config::get<double>("eps_pt_dtm");
//   const double eps_pt_pre_dtm = Config::get<double>("eps_pt_pre_dtm");
//   Timer::start(Util::str_printf("dtm %#.4g", eps_pt_dtm));
//   const size_t n_var_dets = system.get_n_dets();
//   const size_t n_batches = Config::get<size_t>("n_batches_pt_dtm");
//   omp_hash_map<std::string, double> hc_sums_pre;
//   omp_hash_map<std::string, double> hc_sums;
//   const size_t n_procs = Parallel::get_n_procs();
//   const size_t proc_id = Parallel::get_proc_id();
//   std::vector<double> energy_pt_dtm_batches;
//   UncertResult energy_pt_dtm;
//   const auto& str_hasher = std::hash<std::string>();
//   const double target_error = Config::get<double>("target_error", 1.0e-5);

//   for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
//     Timer::start(Util::str_printf("#%zu", batch_id));

//     Timer::start("search");
// #pragma omp parallel for schedule(dynamic, 5)
//     for (size_t i = 0; i < n_var_dets; i++) {
//       const Det& var_det = system.get_det(i);
//       const double coef = system.coefs[i];
//       const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
//         const auto& det_a_code = hps::to_string(det_a);
//         if (var_det_strs.count(det_a_code) == 1) return;
//         const size_t det_a_hash = str_hasher(det_a_code);
//         if (det_a_hash % n_procs != proc_id) return;
//         if ((det_a_hash / n_procs) % n_batches != batch_id) return;
//         const double h_ai = system.get_hamiltonian_elem(var_det, det_a, n_excite);
//         const double hc = h_ai * coef;
//         hc_sums.set(det_a_code, [&](double& value) { value += hc; }, 0.0);
//         if (std::abs(hc) < eps_pt_pre_dtm) return;
//         hc_sums_pre.set(det_a_code, [&](double& value) { value += hc; }, 0.0);
//       };
//       system.find_connected_dets(var_det, Util::INF, eps_pt_dtm / std::abs(coef),
//       pt_det_handler);
//     }
//     if (Parallel::is_master()) {
//       printf("Number of dtm pt dets: %'zu\n", hc_sums.get_n_keys());
//     }
//     Timer::end();  // search

//     Timer::start("accumulate");
//     double energy_pt_dtm_batch = 0.0;
//     hc_sums.apply([&](const std::string& det_a_code, const double hc_sum) {
//       const auto& det_a = hps::from_string<Det>(det_a_code);
//       const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
//       const double hc_sum_pre = hc_sums_pre.get_copy_or_default(det_a_code, 0.0);
//       const double hc_sum_sq_diff = hc_sum * hc_sum - hc_sum_pre * hc_sum_pre;
//       const double contribution = hc_sum_sq_diff / (system.energy_var - H_aa);
// #pragma omp atomic
//       energy_pt_dtm_batch += contribution;
//     });
//     energy_pt_dtm_batches.push_back(energy_pt_dtm_batch);
//     energy_pt_dtm.value = Util::avg(energy_pt_dtm_batches) * n_batches;
//     if (batch_id == n_batches - 1) {
//       energy_pt_dtm.uncert = 0.0;
//     } else {
//       energy_pt_dtm.uncert = Util::stdev(energy_pt_dtm_batches) * n_batches / sqrt(batch_id
//       + 1.0);
//     }
//     if (Parallel::is_master()) {
//       printf("PT dtm correction batch: " ENERGY_FORMAT "\n", energy_pt_dtm_batch);
//       printf("PT dtm correction: %s Ha\n", energy_pt_dtm.to_string().c_str());
//       printf("PT dtm energy: %s Ha\n", (energy_pt_dtm +
//       energy_pt_pre_dtm).to_string().c_str());
//     }
//     Timer::end();  // accumulate

//     hc_sums_pre.clear();
//     hc_sums.clear();
//     Timer::end();  // batch

//     if (batch_id >= 3 && batch_id < n_batches * 0.8 && energy_pt_dtm.uncert < target_error *
//     0.2)
//     {
//       break;
//     }
//   }

//   Timer::end();  // dtm
//   return energy_pt_dtm + energy_pt_pre_dtm;
// }

// template <class S>
// UncertResult Solver<S>::get_energy_pt_sto(const UncertResult& energy_pt_dtm) {
//   const double eps_pt_dtm = Config::get<double>("eps_pt_dtm");
//   const double eps_pt = Config::get<double>("eps_pt");
//   const size_t max_pt_iterations = Config::get<size_t>("max_pt_iterations", 100);
//   omp_hash_map<std::string, double> hc_sums_dtm;
//   omp_hash_map<std::string, double> hc_sums;
//   const size_t n_procs = Parallel::get_n_procs();
//   const size_t proc_id = Parallel::get_proc_id();
//   const size_t n_var_dets = system.get_n_dets();
//   const size_t n_batches = Config::get<size_t>("n_batches_pt_sto");
//   const size_t n_samples = Config::get<size_t>("n_samples_pt_sto");
//   std::vector<double> probs(n_var_dets);
//   std::vector<double> cum_probs(n_var_dets);  // For sampling.
//   std::unordered_map<size_t, unsigned> sample_dets;
//   std::vector<size_t> sample_dets_list;
//   size_t iteration = 0;
//   const auto& str_hasher = std::hash<std::string>();
//   const double target_error = Config::get<double>("target_error", 1.0e-5);
//   UncertResult energy_pt_sto;
//   std::vector<double> energy_pt_sto_loops;

//   // Contruct probs.
//   double sum_weights = 0.0;
//   for (size_t i = 0; i < n_var_dets; i++) sum_weights += std::abs(system.coefs[i]);
//   for (size_t i = 0; i < n_var_dets; i++) {
//     probs[i] = std::abs(system.coefs[i]) / sum_weights;
//     cum_probs[i] = probs[i];
//     if (i > 0) cum_probs[i] += cum_probs[i - 1];
//   }

//   Timer::start(Util::str_printf("sto %#.4g", eps_pt));
//   srand(time(NULL));
//   while (iteration < max_pt_iterations) {
//     Timer::start(Util::str_printf("#%zu", iteration));

//     // Generate random sample
//     for (size_t i = 0; i < n_samples; i++) {
//       const double rand_01 = ((double)rand() / (RAND_MAX));
//       const int sample_det_id =
//           std::lower_bound(cum_probs.begin(), cum_probs.end(), rand_01) - cum_probs.begin();
//       if (sample_dets.count(sample_det_id) == 0) sample_dets_list.push_back(sample_det_id);
//       sample_dets[sample_det_id]++;
//     }

//     // Select random batch.
//     const size_t batch_id = rand() % n_batches;
//     const size_t n_unique_samples = sample_dets_list.size();
//     double energy_pt_sto_loop = 0.0;

//     Timer::start("search");
// #pragma omp parallel for schedule(dynamic, 2)
//     for (size_t sample_id = 0; sample_id < n_unique_samples; sample_id++) {
//       const size_t i = sample_dets_list[sample_id];
//       const double count = static_cast<double>(sample_dets[i]);
//       const double prob = probs[i];
//       const Det& var_det = system.get_det(i);
//       const double coef = system.coefs[i];
//       const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
//         const auto& det_a_code = hps::to_string(det_a);
//         if (var_det_strs.count(det_a_code) == 1) return;
//         const size_t det_a_hash = str_hasher(det_a_code);
//         if (det_a_hash % n_procs != proc_id) return;
//         if ((det_a_hash / n_procs) % n_batches != batch_id) return;
//         const double h_ai = system.get_hamiltonian_elem(var_det, det_a, n_excite);
//         const double hc = h_ai * coef;
//         const double h_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
//         const double factor =
//             n_batches / ((system.energy_var - h_aa) * n_samples * (n_samples - 1));
//         const double contrib_1 = count * hc / prob * sqrt(-factor);
//         hc_sums.set(det_a_code, [&](double& value) { value += contrib_1; }, 0.0);
//         if (std::abs(hc) < eps_pt_dtm) {
//           const double contrib_2 =
//               (count * (n_samples - 1) / prob - (count * count) / (prob * prob)) * hc * hc *
//               factor;
// #pragma omp atomic
//           energy_pt_sto_loop += contrib_2;
//         } else {
//           hc_sums_dtm.set(det_a_code, [&](double& value) { value += contrib_1; }, 0.0);
//         }
//       };
//       system.find_connected_dets(var_det, Util::INF, eps_pt / std::abs(coef),
//       pt_det_handler);
//     }
//     if (Parallel::is_master()) {
//       printf("Number of sto pt dets: %'zu\n", hc_sums.get_n_keys());
//     }
//     Timer::end();
//     sample_dets.clear();
//     sample_dets_list.clear();

//     Timer::start("accumulate");
//     hc_sums.apply([&](const std::string& det_a_code, const double hc_sum) {
//       const double hc_sum_dtm = hc_sums_dtm.get_copy_or_default(det_a_code, 0.0);
//       const double hc_sum_sq_diff = hc_sum * hc_sum - hc_sum_dtm * hc_sum_dtm;
// #pragma omp atomic
//       energy_pt_sto_loop -= hc_sum_sq_diff;
//     });
//     energy_pt_sto_loops.push_back(energy_pt_sto_loop);
//     energy_pt_sto.value = Util::avg(energy_pt_sto_loops);
//     energy_pt_sto.uncert = Util::stdev(energy_pt_sto_loops) / sqrt(iteration + 1.0);
//     if (Parallel::is_master()) {
//       printf("PT sto correction loop: " ENERGY_FORMAT "\n", energy_pt_sto_loop);
//       printf("PT sto correction: %s Ha\n", energy_pt_sto.to_string().c_str());
//       printf("PT sto energy: %s Ha\n", (energy_pt_sto + energy_pt_dtm).to_string().c_str());
//     }
//     Timer::end();

//     hc_sums_dtm.clear();
//     hc_sums.clear();
//     Timer::end();
//     iteration++;
//     if (iteration >= 5 && (energy_pt_sto + energy_pt_dtm).uncert < target_error) {
//       break;
//     }
//   }
//   Timer::end();
//   return energy_pt_sto + energy_pt_dtm;
// }

template <class S>
bool Solver<S>::load_variation_result(const std::string& filename) {
  std::ifstream file(filename, std::ifstream::binary);
  if (!file) return false;
  hps::from_stream<S>(file, system);
  if (Parallel::get_proc_id() == 0) {
    printf("Loaded %'zu dets from: %s\n", system.get_n_dets(), filename.c_str());
    printf("HF energy: " ENERGY_FORMAT "\n", system.energy_hf);
    printf("Variation energy: " ENERGY_FORMAT "\n", system.energy_var);
  }
  return true;
}

template <class S>
void Solver<S>::save_variation_result(const std::string& filename) {
  if (Parallel::get_proc_id() == 0) {
    std::ofstream file(filename, std::ofstream::binary);
    hps::to_stream(system, file);
    printf("Variation results saved to: %s\n", filename.c_str());
  }
}
