#pragma once

#include "../config.h"
#include "../det/det.h"
#include "../result.h"
#include "../timer.h"
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
void Solver<S>::setup() {}

template <class S>
void Solver<S>::run_all_variations() {
  const auto& eps_vars = Config::get<std::vector<double>>("eps_vars");
  for (const double eps_var : eps_vars) {
    Timer::start(str(boost::format("eps_var %#.4g") % eps_var));
    const auto& filename = str(boost::format("var_%#.4g.dat") % eps_var);
    if (!load_variation_result(filename)) {
      run_variation(eps_var);
      save_variation_result(filename);
    }
    Timer::end();
  }
}

template <class S>
void Solver<S>::run_variation(const double eps_var) {
  Det det;
  std::string det_str;
  Davidson davidson;
  const auto& new_det_handler = [&](const Det&) {
    // Serialize to det_str
    system.dets.push_back(det_str);
  };
  size_t n_dets = system.dets.size();
  double energy_var = 0;
  bool converged = false;
  while (!converged) {
    for (size_t i = 0; i < n_dets; i++) {
      // Parse from det_str to det.
      const double coef = system.coefs[i];
      system.find_connected_dets(det, eps_var / std::abs(coef), new_det_handler);
    }
    const size_t n_dets_new = system.dets.size();
    hamiltonian.update(system);
    davidson.diagonalize(hamiltonian.matrix, system.coefs);
    const double energy_var_new = davidson.get_eigenvalue();
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
