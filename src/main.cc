#include <cstdio>
#include <ctime>
#include "config.h"
#include "injector.h"
#include "parallel.h"
#include "result.h"

void print_info(char executable[]) {
  printf("Semistochastic Heat-bath Configuration Interation (SHCI)\n\n");

  printf("Executable: %s\n", executable);

  const time_t start_time = time(0);
  printf("Start time: %s", asctime(localtime(&start_time)));

  const int n_procs = Parallel::get_n_procs();
  const int n_threads = Parallel::get_n_threads();
  printf("Infrastructure: %d nodes * %d threads\n", n_procs, n_threads);
  printf("Master node: %s\n", getenv("HOSTNAME"));
  printf("Configuration:\n");
  Config::print();
}

int main(int, char* argv[]) {
  MPI_Init(nullptr, nullptr);

  if (Parallel::is_master()) print_info(argv[0]);

  Result::init();
  
  // Run the main routine. (Direct Injection pattern)
  Injector::run();

  MPI_Finalize();

  return 0;
}
