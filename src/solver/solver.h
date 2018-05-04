#pragma once

#include <hps/src/hps.h>
#include <omp_hash_map/src/omp_hash_map.h>
#include <cmath>
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

  std::vector<double> eps_prev;

  std::unordered_set<std::string> var_det_strs;

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
  eps_prev.clear();
  var_det_strs.clear();
  auto it_schedule = eps_vars_schedule.begin();
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

template <class S>
void Solver<S>::run_all_perturbations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.4g", eps_var));
    run_perturbation(eps_var);
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var, const bool until_converged) {
  Davidson davidson;
  const auto& connected_det_handler = [&](const Det& connected_det, const double) {
    const auto& det_str = hps::to_string(connected_det);
    if (var_det_strs.count(det_str) == 0) {
      system.det_strs.push_back(det_str);
      system.coefs.push_back(0.0);
      var_det_strs.insert(det_str);
    }
  };
  size_t n_dets = system.get_n_dets();
  double energy_var_prev = 0.0;
  bool converged = false;
  size_t iteration = 0;
  while (!converged) {
    eps_prev.resize(n_dets, Util::INF);
    Timer::start(Util::str_printf("#%zu", iteration));
    for (size_t i = 0; i < n_dets; i++) {
      const double coef = system.coefs[i];
      const double eps_min = eps_var / std::abs(coef);
      if (eps_min >= eps_prev[i]) continue;
      const auto& det = system.get_det(i);
      system.find_connected_dets(det, eps_prev[i], eps_min, connected_det_handler);
      eps_prev[i] = eps_min;
    }
    const size_t n_dets_new = system.get_n_dets();
    if (Parallel::is_master()) {
      printf("Number of dets / new dets: %'zu / %'zu\n", n_dets_new, n_dets_new - n_dets);
    }
    hamiltonian.update(system);
    // for (size_t i = 0; i < n_dets_new; i++) {
    //   printf("diag i: %zu %f\n", i, hamiltonian.matrix.get_diag(i));
    //   if (hamiltonian.matrix.rows[i].size() > 1) {
    //     printf(
    //         "; next %zu %f\n",
    //         hamiltonian.matrix.rows[i].get_index(1),
    //         hamiltonian.matrix.rows[i].get_value(1));
    //   }
    // }
    davidson.diagonalize(hamiltonian.matrix, system.coefs, Parallel::is_master());
    // system.coefs = davidson.get_lowest_eigenvector();
    // hamiltonian.update(system);
    // davidson.diagonalize(hamiltonian.matrix, system.coefs, Parallel::is_master());
    // exit(0);
    const double energy_var_new = davidson.get_lowest_eigenvalue();
    system.coefs = davidson.get_lowest_eigenvector();
    Timer::checkpoint("hamiltonian diagonalized");
    if (Parallel::is_master()) {
      printf("Current variational energy: " ENERGY_FORMAT "\n", energy_var_new);
    }
    if (std::abs(energy_var_new - energy_var_prev) < 1.0e-6) {
      converged = true;
    }
    n_dets = n_dets_new;
    energy_var_prev = energy_var_new;
    Timer::end();
    // if (!until_converged) break;
    iteration++;
  }
  system.energy_var = energy_var_prev;
}

template <class S>
void Solver<S>::run_perturbation(const double eps_var) {
  // If result already exists, return.
  const double eps_pt = Config::get<double>("eps_pt");
  const auto& value_entry = Util::str_printf("energy_pt/%#.4g/%#.4g/value", eps_var, eps_pt);
  const auto& uncert_entry = Util::str_printf("energy_pt/%#.4g/%#.4g/uncert", eps_var, eps_pt);
  UncertResult res(Result::get<double>(value_entry, 0.0));
  if (res.value != 0.0) {
    if (Parallel::is_master()) {
      res.uncert = Result::get<double>(uncert_entry, 0.0);
      printf("PT energy: %s (loaded from result file)\n", res.to_string().c_str());
    }
    // return;
  }

  // Load var wf.
  const auto& var_filename = Util::str_printf("var_%#.4g.dat", eps_var);
  if (!load_variation_result(var_filename)) {
    throw new std::runtime_error("cannot load variation results");
  }

  // Perform multi stage PT.
  var_det_strs.clear();
  for (const auto& det_str : system.det_strs) var_det_strs.insert(det_str);
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
  Timer::start(Util::str_printf("pre dtm %#.4g", eps_pt_pre_dtm));
  const size_t n_var_dets = system.get_n_dets();
  omp_hash_map<std::string, double> hc_sums;

  Timer::start("search");
#pragma omp parallel for schedule(dynamic, 5)
  for (size_t i = 0; i < n_var_dets; i++) {
    const Det var_det = system.get_det(i);
    const double coef = system.coefs[i];
    const auto& pt_det_handler = [&](const Det& det_a, const double h_ai) {
      const auto& det_a_code = hps::to_string(det_a);
      if (var_det_strs.count(det_a_code) == 1) return;
      const double hc = h_ai * coef;
      hc_sums.set(det_a_code, [&](double& value) { value += hc; }, 0.0);
    };
    system.find_connected_dets(var_det, Util::INF, eps_pt_pre_dtm / std::abs(coef), pt_det_handler);
  }
  if (Parallel::is_master()) {
    printf("Number of pre dtm pt dets: %'zu\n", hc_sums.get_n_keys());
  }
  Timer::end();  // search

  Timer::start("accumulate");
  double energy_pt_pre_dtm = 0.0;
  hc_sums.apply([&](const std::string& det_a_code, const double hc_sum) {
    const auto& det_a = hps::from_string<Det>(det_a_code);
    const double H_aa = system.get_hamiltonian_elem(det_a, det_a, 0);
    const double contribution = hc_sum * hc_sum / (system.energy_var - H_aa);
#pragma omp atomic
    energy_pt_pre_dtm += contribution;
  });
  if (Parallel::is_master()) {
    printf("PT pre dtm correction: " ENERGY_FORMAT "\n", energy_pt_pre_dtm);
    printf("PT pre dtm energy: " ENERGY_FORMAT "\n", energy_pt_pre_dtm + system.energy_var);
  }
  Timer::end();  // accumulate

  Timer::end();  // pre dtm
  return energy_pt_pre_dtm;
}

template <class S>
UncertResult Solver<S>::get_energy_pt_dtm(const double energy_pt_pre_dtm) {
  return UncertResult(energy_pt_pre_dtm - 1.0, 1.0);
}

template <class S>
UncertResult Solver<S>::get_energy_pt_sto(const UncertResult& energy_pt_dtm) {
  return energy_pt_dtm + UncertResult(-1.0, 1.0);
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
