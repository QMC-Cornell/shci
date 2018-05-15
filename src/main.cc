#include <csignal>
#include <cstdio>
#include <ctime>
#include "chem/chem_system.h"
#include "config.h"
#include "heg/heg_system.h"
#include "parallel.h"
#include "solver/solver.h"
#include "util.h"

void print_info() {
  printf("Semistochastic Heat-bath Configuration Interation (SHCI)\n\n");
  const time_t start_time = time(0);
  printf("Start time: %s", asctime(localtime(&start_time)));
  const int n_procs = Parallel::get_n_procs();
  const int n_threads = Parallel::get_n_threads();
  printf("Infrastructure: %d nodes * %d threads\n", n_procs, n_threads);
  printf("Configuration:\n");
  Config::print();
}

int main() {
  signal(SIGSEGV, Util::error_handler);
  MPI_Init(nullptr, nullptr);

  if (Parallel::is_master()) print_info();

  const auto& type = Config::get<std::string>("system");
  Parallel::barrier();
  if (type == "heg") {
    Solver<HegSystem>().run();
  } else if (type == "chem") {
    Solver<ChemSystem>().run();
  } else {
    throw std::invalid_argument(Util::str_printf("system '%s' is not supported.", type.c_str()));
  }

  MPI_Finalize();
  return 0;
}
