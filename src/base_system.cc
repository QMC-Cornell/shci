#include "base_system.h"

#include "config.h"
#include "result.h"

void BaseSystem::setup() {
  n_up = Config::get<int>("n_up");
  n_dn = Config::get<int>("n_dn");
  n_elecs = n_up + n_dn;
  Result::put("n_elecs", n_elecs);
}
