#include "injector.h"

#include "chem/chem_system.h"
#include "config.h"
#include "heg/heg_system.h"
#include "parallel.h"
#include "solver/solver.h"
#include "util.h"

void Injector::run() {
  const auto& type = Config::get<std::string>("system");
  Parallel::barrier();
  if (type == "heg") {
    Solver<HegSystem>().run();
  } else if (type == "chem") {
    Solver<ChemSystem>().run();
  } else {
    throw std::invalid_argument(Util::str_printf("system '%s' is not supported.", type.c_str()));
  }
}

