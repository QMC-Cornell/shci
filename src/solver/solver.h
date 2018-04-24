#pragma once

#include <cfloat>
#include <unordered_set>
#include "../config.h"
#include "../det/det.h"
#include "../result.h"
#include "../timer.h"
#include "../util.h"
#include "davidson.h"
#include "hamiltonian.h"

template <class S>
class Solver {
 public:
  void run();

 private:
  S system;

  Hamiltonian<S> hamiltonian;

  void run_all_variations();

  void run_variation(const double eps_var, const bool until_converged = true);

  bool load_variation_result(const std::string& filename);

  void save_variation_result(const std::string& filename);
};

template <class S>
void Solver<S>::run() {
  std::setlocale(LC_ALL, "en_US.UTF-8");
  system.setup();
  Timer::start("variation");
  run_all_variations();
  Timer::end();

  Timer::start("perturbation");
  Timer::end();

  Result::dump();
}

template <class S>
void Solver<S>::run_all_variations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  const auto& eps_vars_schedule = Config::get<std::vector<double>>("eps_vars_schedule");
  double eps_var_prev = Util::INF;
  auto it_schedule = eps_vars_schedule.begin();
  for (const double eps_var : eps_vars) {
    Timer::start(Util::str_printf("eps_var %#.4g", eps_var));
    const auto& filename = Util::str_printf("var_%#.4g.dat", eps_var);
    if (!load_variation_result(filename)) {
      // Perform extra scheduled eps.
      while (it_schedule != eps_vars_schedule.end() && *it_schedule >= eps_var_prev) it_schedule++;
      while (it_schedule != eps_vars_schedule.end() && *it_schedule > eps_var) {
        const double eps_var_extra = *it_schedule;
        Timer::start(Util::str_printf("extra %#.4g", eps_var_extra));
        run_variation(eps_var_extra, false);
        Timer::end();
        it_schedule++;
      }

      Timer::start("main");
      run_variation(eps_var);
      Timer::end();

      save_variation_result(filename);
    }
    eps_var_prev = eps_var;
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var, const bool until_converged) {
  Det tmp_det;
  std::string tmp_det_str;
  Davidson davidson;
  std::unordered_set<std::string> var_dets_set;
  for (const auto& var_det : system.dets) var_dets_set.insert(var_det);
  const auto& connected_det_handler = [&](const Det& connected_det, const double) {
    hps::serialize_to_string(connected_det, tmp_det_str);
    if (var_dets_set.count(tmp_det_str) == 1) return;
    system.dets.push_back(tmp_det_str);
    var_dets_set.insert(tmp_det_str);
  };
  size_t n_dets = system.dets.size();
  double energy_var = 0.0;
  bool converged = false;
  std::vector<double> coefs_prev(n_dets, 0.0);
  while (!converged && until_converged) {
    for (size_t i = 0; i < n_dets; i++) {
      const double coef = system.coefs[i];
      const double coef_prev = coefs_prev[i];
      if (abs(coef) <= abs(coef_prev)) continue;
      hps::parse_from_string(tmp_det, system.dets[i]);
      const double eps_max = coef_prev == 0.0 ? Util::INF : eps_var / abs(coef_prev);
      const double eps_min = eps_var / abs(coef);
      system.find_connected_dets(tmp_det, eps_max, eps_min, connected_det_handler);
    }
    const size_t n_dets_new = system.dets.size();
    hamiltonian.update(system);
    davidson.diagonalize(hamiltonian.matrix, system.coefs);
    const double energy_var_new = davidson.get_eigenvalue();
    coefs_prev = system.coefs;
    system.coefs = davidson.get_eigenvector();
    if (n_dets_new == n_dets && abs(energy_var_new - energy_var) < 1.0e-6) {
      converged = true;
    }
    n_dets = n_dets_new;
    energy_var = energy_var_new;
  }
  system.energy_var = energy_var;
}

template <class S>
bool Solver<S>::load_variation_result(const std::string& filename) {
  std::ifstream file(filename, std::ifstream::binary);
  if (!file) return false;
  hps::parse_from_stream<S>(system, file);
  if (Parallel::get_proc_id() == 0) {
    printf("Loaded %'zu dets from: %s\n", system.dets.size(), filename.c_str());
    printf("HF energy: " ENERGY_FORMAT "\n", system.energy_hf);
    printf("Variation energy: " ENERGY_FORMAT "\n", system.energy_var);
  }
  return true;
}

template <class S>
void Solver<S>::save_variation_result(const std::string& filename) {
  if (Parallel::get_proc_id() == 0) {
    std::ofstream file(filename, std::ofstream::binary);
    hps::serialize_to_stream(system, file);
    printf("Variation results saved to: %s\n", filename.c_str());
  }
}
