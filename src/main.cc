#include <boost/format.hpp>
#include <cstdio>
#include <ctime>
#include "chem/chem_system.h"
#include "config.h"
#include "parallel.h"
#include "solver/solver.h"

int main() {
  const int proc_id = Parallel::get_proc_id();
  if (proc_id == 0) {
    printf("Semistochastic Heat-bath Configuration Interation (SHCI)\n\n");
    const time_t start_time = time(0);
    printf("Start time: %s", asctime(localtime(&start_time)));
    const int n_procs = Parallel::get_n_procs();
    const int n_threads = Parallel::get_n_threads();
    printf("Infrastructure: %d nodes * %d threads\n", n_procs, n_threads);
    printf("Configuration:\n");
    Config::print();
  }

  const auto& type = Config::get<std::string>("system");
  if (type == "heg") {
    // TODO.
  } else if (type == "chem") {
    Solver<ChemSystem>().run();
  } else {
    throw std::invalid_argument(str(boost::format("system '%s' not supported.") % type.c_str()));
  }

  return 0;
}
