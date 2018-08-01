#include "../base_system.h"
#include "hamiltonian.h"

template <class S>
class Green {
 public:
  Green(const S& system, const Hamiltonian<S>& hamiltonian)
      : system(system), hamiltonian(hamiltonian) {}

  void run();

 private:
  const S& system;

  const Hamiltonian<S>& hamiltonian;

  std::vector<double> construct_b(const unsigned orb);
};
