#include "../base_system.h"
#include "hamiltonian.h"

#include <cstdio>
#include <unordered_map>
#include <vector>
#include "../util.h"

template <class S>
class Green {
 public:
  Green(S& system, Hamiltonian<S>& hamiltonian) : system(system), hamiltonian(hamiltonian) {}

  void run();

 private:
  size_t n_dets;

  size_t n_pdets;

  unsigned n_orbs;

  double w;

  double n;

  bool advanced;

  std::vector<Det> dets_store;

  std::vector<double> coefs_store;

  std::unordered_map<Det, size_t, DetHasher> pdet_to_id;

  S& system;

  Hamiltonian<S>& hamiltonian;

  std::vector<std::vector<std::complex<double>>> G;

  void construct_pdets();

  std::vector<double> construct_b(const unsigned orb);

  std::vector<std::complex<double>> mul_green(const std::vector<std::complex<double>>& vec) const;

  std::vector<std::complex<double>> cg(
      const std::vector<double>& b,
      const std::vector<std::complex<double>>& x0,
      const double tol = 1.0e-15);

  void output_green();
};

template <class S>
void Green<S>::run() {
  // Store dets and coefs.
  dets_store = system.dets;
  coefs_store = system.coefs[0];
  n_dets = dets_store.size();
  n_orbs = system.n_orbs;

  w = Config::get<double>("w_green");
  n = Config::get<double>("n_green");

  // Construct new dets.
  system.dets.clear();
  system.coefs[0].clear();
  advanced = Config::get<bool>("advanced_green", false);
  if (Parallel::is_master()) {
    if (advanced) {
      printf("Calculating G-\n");
    } else {
      printf("Calculating G+\n");
    }
  }
  construct_pdets();

  // Construct hamiltonian.
  hamiltonian.clear();
  hamiltonian.update(system);

  // Initialize G.
  G.resize(n_orbs * 2);
  for (unsigned i = 0; i < n_orbs * 2; i++) {
    G[i].assign(n_orbs * 2, 0.0);
  }

  for (unsigned j = 0; j < n_orbs * 2; j++) {
    Timer::checkpoint(Util::str_printf("orb #%zu/%zu", j + 1, n_orbs * 2));
    // Construct bj
    const auto& bj = construct_b(j);

    // Generate initial x0.
    std::vector<std::complex<double>> x0(n_pdets, 1.0e-6);

    for (size_t k = 0; k < n_pdets; k++) {
      if (std::abs(bj[k]) > 1.0e-6) {
        std::complex<double> diag = hamiltonian.matrix.get_diag(k);
        if (advanced) {
          diag = w + n * Util::I - (diag - system.energy_var[0]);
        } else {
          diag = w + n * Util::I + (diag - system.energy_var[0]);
        }
        x0[k] = bj[k] / diag;
      }
    }

    // Iteratively get H^{-1}bj
    const auto& x = cg(bj, x0);

    for (unsigned i = 0; i < n_orbs * 2; i++) {
      // Dot with bi
      const auto& bi = construct_b(i);
      G[i][j] = Util::dot_omp(bi, x);
    }
  }

  output_green();
}

template <class S>
void Green<S>::construct_pdets() {
  for (size_t i = 0; i < n_dets; i++) {
    Det det = dets_store[i];
    for (unsigned k = 0; k < n_orbs; k++) {
      if (advanced) {  // G-.
        if (det.up.has(k)) {
          det.up.unset(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.up.set(k);
        }
        if (det.dn.has(k)) {
          det.dn.unset(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.dn.set(k);
        }
      } else {  // G+.
        if (!det.up.has(k)) {
          det.up.set(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.up.unset(k);
        }
        if (!det.dn.has(k)) {
          det.dn.set(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.dn.unset(k);
        }
      }  // Advanced.
    }
  }
  n_pdets = system.dets.size();
  system.coefs[0].assign(n_pdets, 0.0);
}

template <class S>
std::vector<double> Green<S>::construct_b(const unsigned j) {
  std::vector<double> b(n_pdets, 0.0);
  for (size_t det_id = 0; det_id < n_dets; det_id++) {
    Det det = dets_store[det_id];
    if (advanced) {  // G-.
      if (j < n_orbs && det.up.has(j)) {
        det.up.unset(j);
      } else if (j >= n_orbs && det.dn.has(j - n_orbs)) {
        det.dn.unset(j - n_orbs);
      } else {
        continue;
      }
    } else {  // G+.
      if (j < n_orbs && !det.up.has(j)) {
        det.up.set(j);
      } else if (j >= n_orbs && !det.dn.has(j - n_orbs)) {
        det.dn.set(j - n_orbs);
      } else {
        continue;
      }
    }  // Advanced.
    const size_t pdet_id = pdet_to_id[det];
    const double coef = coefs_store[det_id];
    b[pdet_id] = coef;
  }
  return b;
}

template <class S>
void Green<S>::output_green() {
  const auto& filename = Util::str_printf("green_%#.2e_%#.2ei.csv", w, n);

  FILE* file = fopen(filename.c_str(), "w");

  fprintf(file, "i,j,G\n");
  for (unsigned i = 0; i < n_orbs * 2; i++) {
    for (unsigned j = 0; j < n_orbs * 2; j++) {
      fprintf(file, "%u,%u,%+.10f%+.10fj\n", i, j, G[i][j].real(), G[i][j].imag());
    }
  }

  fclose(file);

  printf("Green's function saved to: %s\n", filename.c_str());
}

template <class S>
std::vector<std::complex<double>> Green<S>::cg(
    const std::vector<double>& b, const std::vector<std::complex<double>>& x0, const double tol) {
  std::vector<std::complex<double>> x(n_pdets, 0.0);
  std::vector<std::complex<double>> r(n_pdets, 0.0);
  std::vector<std::complex<double>> p(n_pdets, 0.0);

  const auto& Ax0 = mul_green(x0);

#pragma omp parallel for
  for (size_t i = 0; i < n_pdets; i++) {
    r[i] = b[i] - Ax0[i];
  }
  p = r;
  x = x0;

  double residual = 1.0;
  int iter = 0;
  while (residual > tol) {
    const std::complex<double>& rTr = Util::dot_omp(r, r);
    const auto& Ap = mul_green(p);
    const std::complex<double>& pTAp = Util::dot_omp(p, Ap);
    const std::complex<double>& a = rTr / pTAp;
#pragma omp parallel for
    for (size_t j = 0; j < n_pdets; j++) {
      x[j] += a * p[j];
      r[j] -= a * Ap[j];
    }
    const std::complex<double>& rTr_new = Util::dot_omp(r, r);
    const std::complex<double>& beta = rTr_new / rTr;
#pragma omp parallel for
    for (size_t j = 0; j < n_pdets; j++) {
      p[j] = r[j] + beta * p[j];
    }

    residual = std::abs(rTr);
    iter++;
    if (iter % 10 == 0) printf("Iteration %d: r = %g\n", iter, residual);
    if (iter > 100) throw std::runtime_error("cg does not converge");
  }

  printf("Final iteration %d: r = %g\n", iter, residual);

  return x;
}

template <class S>
std::vector<std::complex<double>> Green<S>::mul_green(
    const std::vector<std::complex<double>>& vec) const {
  auto G_vec = hamiltonian.matrix.mul(vec);

  for (size_t i = 0; i < n_pdets; i++) {
    if (advanced) {
      G_vec[i] = (w + n * Util::I + system.energy_var[0]) * vec[i] - G_vec[i];
    } else {
      G_vec[i] = (w + n * Util::I - system.energy_var[0]) * vec[i] + G_vec[i];
    }
  }

  return G_vec;
}
