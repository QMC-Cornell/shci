#pragma once

#include "../system.h"
#include "../timer.h"

class Solver {
 public:
  void set_system(System* sys) { this->sys = sys; }

  void solve() {
    Timer::start("variation");
    Timer::end();

    Timer::start("perturbation");
    Timer::end();
  }

 private:
  System* sys;
};
