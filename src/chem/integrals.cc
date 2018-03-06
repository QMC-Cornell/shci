#include "integrals.h"

#include "../parallel.h"

void Integrals::load_fcidump() {
  FILE* fcidump = fopen("FCIDUMP", "r");
  if (!fcidump) throw new std::runtime_error("FCIDUMP not found");
  char buf[128];
  fscanf(fcidump, " %*4s %*5s %u %*7s %u %*s", &n_orbs, &n_elecs);
  printf("Reading FCIDUMP...\n");
  printf("n_orbs: %u, n_elecs: %u\n", n_orbs, n_elecs);
}
