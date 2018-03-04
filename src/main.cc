#include <cstdio>
#include <ctime>
#include "config.h"
#include "parallel.h"

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

  return 0;
}
