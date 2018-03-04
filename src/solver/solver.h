#pragma once

#include "../config.h"
#include "../result.h"
#include "../timer.h"
#include "sparse_hamiltonian.h"

template <class S>
class Solver {
 public:
  void solve();

 private:
  S system;

  SparseHamiltonian<S> ham;

  void setup();
};

template <class S>
void Solver<S>::solve() {
  system.setup();

  Timer::start("variation");
  system.setup_variation();
  Timer::end();

  Timer::start("perturbation");
  system.setup_perturbation();
  Timer::end();

  Result::dump();
}
