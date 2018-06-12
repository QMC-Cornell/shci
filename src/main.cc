#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <thread>
#include "config.h"
#include "injector.h"
#include "parallel.h"
#include "result.h"
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
  Result::init();
  Parallel::barrier();
  Result::dump();

  Injector::run();

  MPI_Finalize();
  return 0;
}
