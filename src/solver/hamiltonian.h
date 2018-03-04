#pragma once

template <class S>
class Hamiltonian {
 public:
  SparseMatrix matrix;

  void update(const S& system);
};

template <class S>
void Hamiltonian<S>::update(const S&) {}
