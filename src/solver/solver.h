#pragma once

#include <unordered_set>
#include "../config.h"
#include "../det/det.h"
#include "../result.h"
#include "../timer.h"
#include "../util/str_printf.h"
#include "davidson.h"
#include "hamiltonian.h"

template <class S>
class Solver {
 public:
  void run();

 private:
  S system;

  Hamiltonian<S> hamiltonian;

  void setup();

  void run_all_variations();

  void run_variation(const double eps_var);

  bool load_variation_result(const std::string& filename);

  void save_variation_result(const std::string& filename);
};

template <class S>
void Solver<S>::run() {
  system.setup();
  setup();
  Timer::start("variation");
  system.setup_variation();
  run_all_variations();
  Timer::end();

  Timer::start("perturbation");
  system.setup_perturbation();
  Timer::end();

  Result::dump();
}

template <class S>
void Solver<S>::setup() {
  std::setlocale(LC_ALL, "en_US.UTF-8");
}

template <class S>
void Solver<S>::run_all_variations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  for (const double eps_var : eps_vars) {
    // Timer::start(str(boost::format("eps_var %#.4g") % eps_var));
    Timer::start(util::str_printf("eps_var %#.4g", eps_var));
    const auto& filename = util::str_printf("var_%#.4g.dat", eps_var);
    if (!load_variation_result(filename)) {
      run_variation(eps_var);
      save_variation_result(filename);
    }
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var) {
  Det tmp_det;
  std::string tmp_det_str;
  Davidson davidson;
  std::unordered_set<std::string> var_dets_set;
  for (const auto& var_det : system.dets) var_dets_set.insert(var_det);
  const auto& connected_det_handler = [&](const Det& connected_det) {
    hps::serialize_to_string(connected_det, tmp_det_str);
    if (var_dets_set.count(tmp_det_str) == 1) return;
    system.dets.push_back(tmp_det_str);
    var_dets_set.insert(tmp_det_str);
  };
  size_t n_dets = system.dets.size();
  double energy_var = 0;
  bool converged = false;
  std::vector<double> coefs_prev(n_dets, 0);
  while (!converged) {
    for (size_t i = 0; i < n_dets; i++) {
      const double coef = system.coefs[i];
      if (std::abs(coef) <= std::abs(coefs_prev[i])) continue;
      hps::parse_from_string(tmp_det, system.dets[i]);
      system.find_connected_dets(tmp_det, eps_var / std::abs(coef), connected_det_handler);
    }
    const size_t n_dets_new = system.dets.size();
    hamiltonian.update(system);
    davidson.diagonalize(hamiltonian.matrix, system.coefs);
    const double energy_var_new = davidson.get_eigenvalue();
    coefs_prev = system.coefs;
    system.coefs = davidson.get_eigenvector();
    if (n_dets_new == n_dets && std::abs(energy_var_new - energy_var) < 1.0e-6) {
      converged = true;
    }
    n_dets = n_dets_new;
    energy_var = energy_var_new;
  }
}

template <class S>
bool Solver<S>::load_variation_result(const std::string&) {
  return false;
}

template <class S>
void Solver<S>::save_variation_result(const std::string&) {}
