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
#include <unordered_set>
#include "../config.h"
#include "../det/det.h"
#include "../math_vector.h"
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

  // std::unordered_set<Det, DetHasher> var_dets;
  fgpl::HashSet<Det, DetHasher> var_dets;

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

  std::string get_wf_filename(const double eps) const;

  template <class C>
  double mapreduce_sum(
      const fgpl::DistHashMap<Det, C, DetHasher>& map,
      const std::function<double(const Det& det, const C& hc_sum)>& mapper) const;
};

template <class S>
void Solver<S>::run() {
  Timer::start("setup");
  std::setlocale(LC_ALL, "en_US.UTF-8");
  system.setup();
  Result::put("energy_hf", system.energy_hf);
  Timer::end();

  Timer::start("variation");
  run_all_variations();
  hamiltonian.clear();
  Timer::end();

  if (Config::get<bool>("var_only", false)) return;

  Timer::start("perturbation");
  run_all_perturbations();
  Timer::end();

  Result::dump();
}

template <class S>
void Solver<S>::run_all_variations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  const auto& eps_vars_schedule = Config::get<std::vector<double>>("eps_vars_schedule");
  double eps_var_prev = Util::INF;
  // for (const auto& det : system.dets) var_dets.insert(det);
  for (const auto& det : system.dets) var_dets.set(det);
  auto it_schedule = eps_vars_schedule.begin();
  var_iteration_global = 0;
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.2e", eps_var));
    const auto& filename = get_wf_filename(eps_var);
    if (!load_variation_result(filename)) {
      // Perform extra scheduled eps.
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var_prev) it_schedule++;
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
      hamiltonian.clear();
    }
    eps_var_prev = eps_var;
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_all_perturbations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
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
      if (eps_min >= eps_tried_prev[i] * 0.99) return;
      const auto& det = system.dets[i];
      const auto& connected_det_handler = [&](const Det& connected_det, const int n_excite) {
        // if (var_dets.count(connected_det) == 1) return;
        if (var_dets.has(connected_det)) return;
        if (n_excite == 1) {
          const double h_ai = system.get_hamiltonian_elem(det, connected_det, n_excite);
          if (std::abs(h_ai) < eps_min) return;  // Filter out small single excitation.
        }
        dist_new_dets.async_set(connected_det);
      };
      system.find_connected_dets(det, eps_tried_prev[i], eps_min, connected_det_handler);
      eps_tried_prev[i] = eps_min;
    });

    dist_new_dets.sync();
    dist_new_dets.for_each_serial([&](const Det& connected_det, const size_t) {
      // var_dets.insert(connected_det);
      var_dets.set(connected_det);
      system.dets.push_back(connected_det);
      system.coefs.push_back(0.0);
    });
    dist_new_dets.clear();

    const size_t n_dets_new = system.coefs.size();
    if (Parallel::is_master()) {
      printf("Number of dets / new dets: %'zu / %'zu\n", n_dets_new, n_dets_new - n_dets);
    }
    Timer::checkpoint("get next det list");

    hamiltonian.update(system);
    davidson.diagonalize(hamiltonian.matrix, system.coefs, Parallel::is_master(), until_converged);
    const double energy_var_new = davidson.get_lowest_eigenvalue();
    system.coefs = davidson.get_lowest_eigenvector();
    Timer::checkpoint("diagonalize sparse hamiltonian");
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

template <class S>
void Solver<S>::run_perturbation(const double eps_var) {
  // If result already exists, return.
  const double eps_pt = Config::get<double>("eps_pt");
  const auto& value_entry = Util::str_printf("energy_pt/%#.2e/%#.2e/value", eps_var, eps_pt);
  const auto& uncert_entry = Util::str_printf("energy_pt/%#.2e/%#.2e/uncert", eps_var, eps_pt);
  UncertResult res(Result::get<double>(value_entry, 0.0));
  if (res.value != 0.0) {
    if (Parallel::is_master()) {
      res.uncert = Result::get<double>(uncert_entry, 0.0);
      printf("PT energy: %s (loaded from result file)\n", res.to_string().c_str());
    }
    if (!Config::get<bool>("force_pt", false)) return;
  }

  // Load var wf.
  const auto& var_filename = get_wf_filename(eps_var);
  if (!load_variation_result(var_filename)) {
    throw new std::runtime_error("cannot load variation results");
  }
  system.update_diag_helper();

  // Perform multi stage PT.
  var_dets.clear();
  // for (const auto& det : system.dets) var_dets.insert(det);
  for (const auto& det : system.dets) var_dets.set(det);
  const double energy_pt_pre_dtm = get_energy_pt_pre_dtm();
  const UncertResult energy_pt_dtm = get_energy_pt_dtm(energy_pt_pre_dtm);
  const UncertResult energy_pt = get_energy_pt_sto(energy_pt_dtm);
  if (Parallel::is_master()) {
    printf("PT energy: %s Ha\n", energy_pt.to_string().c_str());
  }
  Result::put(value_entry, energy_pt.value);
  Result::put(uncert_entry, energy_pt.uncert);
}

template <class S>
double Solver<S>::get_energy_pt_pre_dtm() {
  const double eps_pt_pre_dtm = Config::get<double>("eps_pt_pre_dtm");
  Timer::start(Util::str_printf("pre dtm %#.2e", eps_pt_pre_dtm));
  const size_t n_var_dets = system.get_n_dets();
  fgpl::DistHashMap<Det, MathVector<double, 1>, DetHasher> hc_sums;

  fgpl::DistRange<size_t>(0, n_var_dets).for_each([&](const size_t i) {
    const Det& det = system.dets[i];
    const double coef = system.coefs[i];
    const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
      // if (var_dets.count(det_a) == 1) return;
      if (var_dets.has(det_a)) return;
      const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
      const double hc = h_ai * coef;
      if (std::abs(hc) < eps_pt_pre_dtm) return;  // Filter out small single excitation.
      const MathVector<double, 1> contrib(hc);
      hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 1>>::sum);
    };
    system.find_connected_dets(det, Util::INF, eps_pt_pre_dtm / std::abs(coef), pt_det_handler);
  });
  hc_sums.sync(fgpl::Reducer<MathVector<double, 1>>::sum);
  const size_t n_pt_dets = hc_sums.get_n_keys();
  if (Parallel::is_master()) {
    printf("Number of pre dtm pt dets: %'zu\n", n_pt_dets);
  }
  Timer::checkpoint("create hc sums");

  double energy_pt_pre_dtm = mapreduce_sum<MathVector<double, 1>>(
      hc_sums, [&](const Det& det_a, const MathVector<double, 1>& hc_sum) {
        const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
        const double contrib = hc_sum[0] * hc_sum[0] / (system.energy_var - H_aa);
        return contrib;
      });
  if (Parallel::is_master()) {
    printf("PT pre dtm correction: " ENERGY_FORMAT "\n", energy_pt_pre_dtm);
    printf("PT pre dtm energy: " ENERGY_FORMAT "\n", energy_pt_pre_dtm + system.energy_var);
  }

  Timer::end();  // pre dtm
  return energy_pt_pre_dtm + system.energy_var;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_dtm(const double energy_pt_pre_dtm) {
  const double eps_pt_dtm = Config::get<double>("eps_pt_dtm");
  const double eps_pt_pre_dtm = Config::get<double>("eps_pt_pre_dtm");
  Timer::start(Util::str_printf("dtm %#.2e", eps_pt_dtm));
  const size_t n_var_dets = system.get_n_dets();
  const size_t n_batches = Config::get<size_t>("n_batches_pt_dtm", 5);
  // fgpl::DistHashMap<Det, double, DetHasher> hc_sums_pre;
  fgpl::DistHashMap<Det, MathVector<double, 2>, DetHasher> hc_sums;
  std::vector<double> energy_pt_dtm_batches;
  UncertResult energy_pt_dtm;
  const DetHasher det_hasher;
  const double target_error = Config::get<double>("target_error", 1.0e-5);

  for (size_t batch_id = 0; batch_id < n_batches; batch_id++) {
    Timer::start(Util::str_printf("#%zu", batch_id));

    fgpl::DistRange<size_t>(0, n_var_dets).for_each([&](const size_t i) {
      const Det& det = system.dets[i];
      const double coef = system.coefs[i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        // if (var_dets.count(det_a) == 1) return;
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if (batch_hash % n_batches != batch_id) return;
        const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
        const double hc = h_ai * coef;
        if (std::abs(hc) < eps_pt_dtm) return;  // Filter out small single excitation.
        MathVector<double, 2> contrib;
        contrib[0] = hc;
        if (std::abs(hc) >= eps_pt_pre_dtm) contrib[1] = hc;
        hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 2>>::sum);
        // hc_sums_pre.async_set(det_a, hc, fgpl::Reducer<double>::sum);
      };
      system.find_connected_dets(det, Util::INF, eps_pt_dtm / std::abs(coef), pt_det_handler);
    });
    hc_sums.sync(fgpl::Reducer<MathVector<double, 2>>::sum);
    // hc_sums_pre.sync(fgpl::Reducer<double>::sum);
    const size_t n_pt_dets = hc_sums.get_n_keys();
    if (Parallel::is_master()) {
      printf("Number of dtm pt dets: %'zu\n", n_pt_dets);
    }
    Timer::checkpoint("create hc sums");

    const double energy_pt_dtm_batch = mapreduce_sum<MathVector<double, 2>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 2>& hc_sum) {
          // const double hc_sum_pre = hc_sums_pre.get_local(det_a, 0.0);
          // TODO: 0.1 ~ 1% difference whether accumulating separately or not.
          const double hc_sum_sq_diff = hc_sum[0] * hc_sum[0] - hc_sum[1] * hc_sum[1];
          const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double contrib = hc_sum_sq_diff / (system.energy_var - H_aa);
          return contrib;
        });
    energy_pt_dtm_batches.push_back(energy_pt_dtm_batch);
    energy_pt_dtm.value = Util::avg(energy_pt_dtm_batches) * n_batches;
    if (batch_id == n_batches - 1) {
      energy_pt_dtm.uncert = 0.0;
    } else {
      energy_pt_dtm.uncert = Util::stdev(energy_pt_dtm_batches) * n_batches / sqrt(batch_id + 1.0);
    }

    if (Parallel::is_master()) {
      printf("PT dtm correction batch: " ENERGY_FORMAT "\n", energy_pt_dtm_batch);
      printf("PT dtm correction: %s Ha\n", energy_pt_dtm.to_string().c_str());
      printf("PT dtm energy: %s Ha\n", (energy_pt_dtm + energy_pt_pre_dtm).to_string().c_str());
    }

    // hc_sums_pre.clear();
    hc_sums.clear();
    Timer::end();  // batch

    if (batch_id >= 3 && batch_id < n_batches * 0.8 && energy_pt_dtm.uncert < target_error * 0.2) {
      break;
    }
  }

  Timer::end();  // dtm
  return energy_pt_dtm + energy_pt_pre_dtm;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_sto(const UncertResult& energy_pt_dtm) {
  const double eps_pt_dtm = Config::get<double>("eps_pt_dtm");
  const double eps_pt = Config::get<double>("eps_pt");
  const size_t max_pt_iterations = Config::get<size_t>("max_pt_iterations", 50);
  // fgpl::DistHashMap<Det, double, DetHasher> hc_sums_dtm;
  fgpl::DistHashMap<Det, MathVector<double, 3>, DetHasher> hc_sums;
  const size_t n_var_dets = system.get_n_dets();
  const size_t n_batches = Config::get<size_t>("n_batches_pt_sto", 5);
  const size_t n_samples = Config::get<size_t>("n_samples_pt_sto", 1000);
  std::vector<double> probs(n_var_dets);
  std::vector<double> cum_probs(n_var_dets);  // For sampling.
  std::unordered_map<size_t, unsigned> sample_dets;
  std::vector<size_t> sample_dets_list;
  size_t iteration = 0;
  const DetHasher det_hasher;
  const double target_error = Config::get<double>("target_error", 1.0e-5);
  UncertResult energy_pt_sto;
  std::vector<double> energy_pt_sto_loops;

  // Contruct probs.
  double sum_weights = 0.0;
  for (size_t i = 0; i < n_var_dets; i++) sum_weights += std::abs(system.coefs[i]);
  for (size_t i = 0; i < n_var_dets; i++) {
    probs[i] = std::abs(system.coefs[i]) / sum_weights;
    cum_probs[i] = probs[i];
    if (i > 0) cum_probs[i] += cum_probs[i - 1];
  }

  srand(time(nullptr));
  Timer::start(Util::str_printf("sto %#.2e", eps_pt));
  while (iteration < max_pt_iterations) {
    Timer::start(Util::str_printf("#%zu", iteration));

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

    // Select random batch.
    size_t batch_id = rand() % n_batches;
    fgpl::broadcast(batch_id);
    const size_t n_unique_samples = sample_dets_list.size();
    if (Parallel::is_master()) {
      printf("Batch id: %zu / %zu\n", batch_id, n_batches);
    }
    // double energy_pt_sto_loop_local = 0.0;
    // std::vector<double> energy_pt_sto_loop_thread(n_threads, 0.0);

    fgpl::DistRange<size_t>(0, n_unique_samples).for_each([&](const size_t sample_id) {
      const size_t i = sample_dets_list[sample_id];
      const double count = static_cast<double>(sample_dets[i]);
      const Det& det = system.dets[i];
      const double coef = system.coefs[i];
      const double prob = probs[i];
      const auto& pt_det_handler = [&](const Det& det_a, const int n_excite) {
        // if (var_dets.count(det_a) == 1) return;
        if (var_dets.has(det_a)) return;
        const size_t det_a_hash = det_hasher(det_a);
        const size_t batch_hash = Util::rehash(det_a_hash);
        if (batch_hash % n_batches != batch_id) return;
        const double h_ai = system.get_hamiltonian_elem(det, det_a, n_excite);
        const double hc = h_ai * coef;
        if (std::abs(hc) < eps_pt) return;  // Filter out small single excitation.
        // const double h_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
        // const double factor =
        //     n_batches / ((system.energy_var - h_aa) * n_samples * (n_samples - 1));
        const double factor = static_cast<double>(n_batches) / (n_samples * (n_samples - 1));
        // const double contrib_1 = count * hc / prob * sqrt(-factor);
        MathVector<double, 3> contrib;
        contrib[0] = count * hc / prob * sqrt(factor);
        if (std::abs(hc) < eps_pt_dtm) {
          // const double contrib_2 =
          //     (count * (n_samples - 1) / prob - (count * count) / (prob * prob)) * hc * hc *
          //     factor;
          // const int thread_id = omp_get_thread_num();
          // energy_pt_sto_loop_thread[thread_id] += contrib_2;
          contrib[2] =
              (count * (n_samples - 1) / prob - (count * count) / (prob * prob)) * hc * hc * factor;
        } else {
          contrib[1] = contrib[0];
          // hc_sums_dtm.async_set(det_a, contrib_1, fgpl::Reducer<double>::sum);
        }
        hc_sums.async_set(det_a, contrib, fgpl::Reducer<MathVector<double, 3>>::sum);
      };
      system.find_connected_dets(det, Util::INF, eps_pt / std::abs(coef), pt_det_handler);
    });
    hc_sums.sync(fgpl::Reducer<MathVector<double, 3>>::sum);
    // hc_sums_dtm.sync(fgpl::Reducer<double>::sum);
    const size_t n_pt_dets = hc_sums.get_n_keys();
    // const size_t n_dtm_pt_dets = hc_sums_dtm.get_n_keys();
    if (Parallel::is_master()) {
      printf("Number of sto pt dets: %'zu\n", n_pt_dets);
      // printf("Number of sto pt dets within eps_pt_dtm: %'zu\n", n_dtm_pt_dets);
    }
    sample_dets.clear();
    sample_dets_list.clear();
    Timer::checkpoint("create hc sums");

    // double energy_pt_sto_loop = 0.0;
    // for (int i = 0; i < n_threads; i++) energy_pt_sto_loop_local += energy_pt_sto_loop_thread[i];
    // MPI_Allreduce(
    //     &energy_pt_sto_loop_local, &energy_pt_sto_loop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double energy_pt_sto_loop = mapreduce_sum<MathVector<double, 3>>(
        hc_sums, [&](const Det& det_a, const MathVector<double, 3>& hc_sum) {
          const double h_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
          const double factor = 1.0 / (system.energy_var - h_aa);
          return (hc_sum[0] * hc_sum[0] - hc_sum[1] * hc_sum[1] + hc_sum[2]) * factor;
          // const double hc_sum_dtm = hc_sums_dtm.get_local(det_a, 0.0);
          // const double hc_sum_sq_diff = hc_sum * hc_sum - hc_sum_dtm * hc_sum_dtm;
          // return hc_sum_sq_diff;
        });
    // energy_pt_sto_loop -= sq_diff;

    energy_pt_sto_loops.push_back(energy_pt_sto_loop);
    energy_pt_sto.value = Util::avg(energy_pt_sto_loops);
    energy_pt_sto.uncert = Util::stdev(energy_pt_sto_loops) / sqrt(iteration + 1.0);
    if (Parallel::is_master()) {
      printf("PT sto correction loop: " ENERGY_FORMAT "\n", energy_pt_sto_loop);
      printf("PT sto correction: %s Ha\n", energy_pt_sto.to_string().c_str());
      printf("PT sto energy: %s Ha\n", (energy_pt_sto + energy_pt_dtm).to_string().c_str());
    }

    // hc_sums_dtm.clear();
    hc_sums.clear();
    Timer::end();
    iteration++;
    if (iteration >= 5 && (energy_pt_sto + energy_pt_dtm).uncert < target_error) {
      break;
    }
  }
  Timer::end();
  return energy_pt_sto + energy_pt_dtm;
}

template <class S>
template <class C>
double Solver<S>::mapreduce_sum(
    const fgpl::DistHashMap<Det, C, DetHasher>& map,
    const std::function<double(const Det& det, const C& hc_sum)>& mapper) const {
  const int n_threads = omp_get_max_threads();
  std::vector<double> res_thread(n_threads, 0.0);
  map.for_each([&](const Det& key, const size_t, const C& value) {
    const int thread_id = omp_get_thread_num();
    res_thread[thread_id] += mapper(key, value);
  });
  double res_local;
  double res;
  res_local = res_thread[0];
  for (int i = 1; i < n_threads; i++) res_local += res_thread[i];
  MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return res;
}

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

template <class S>
std::string Solver<S>::get_wf_filename(const double eps) const {
  return Util::str_printf("wf_eps1_%#.2e.dat", eps);
}
