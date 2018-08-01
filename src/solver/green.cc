#include "green.h"

#include <unordered_map>


std::vector<double> Green::construct_b(const unsigned orb) {
}

/**
void Green::get_green(
    const std::vector<Det>& dets,
    const std::vector<double>& coefs,
    const Integrals& integrals,
    const unsigned n_orbs,
    const double w) {
  std::vector<Det> det_p1s;
  std::unordered_map<Det, size_t, DetHasher> det_p1_to_id;
  Timer::start("construct det+1");
  for (const auto& det : dets) {
    Det det_copy = det;
    for (unsigned i = 0; i < n_orbs; i++) {
      if (!det.up.has(i)) {
        det_copy.up.set(i);
        if (det_p1_to_id.count(det_copy) == 0) {
          det_p1_to_id[det_copy] = det_p1s.size();
          det_p1s.push_back(det_copy);
        }
        det_copy.up.unset(i);
      }
      if (!det.dn.has(i)) {
        det_copy.dn.set(i);
        if (det_p1_to_id.count(det_copy) == 0) {
          det_p1_to_id[det_copy] = det_p1s.size();
          det_p1s.push_back(det_copy);
        }
        det_copy.dn.unset(i);
      }
    }
  }
  const size_t n_dets_p1 = det_p1s.size();
  Timer::end();
  // GreenMatrix green_matrix;
  // green_matrix.construct(det_p1s);

  std::vector<std::vector<double>> res;

  for (unsigned j = 0; j < n_orbs * 2; j++) {
    // Construct b_j.
    std::vector<double> b_j(n_dets_p1, 0.0);
    for (const size_t det_id = 0; det_id < dets.size(); det_id++) {
      Det det_copy = dets[det_id];
      const double coef = coefs[det_id];
      if (j < n_orbs && !det.up.has(j)) {
        det_copy.up.set(j);
        const size_t det_p1_id = det_p1_to_id[det_copy];
        b_j[det_p1_id] = coef;
      } else if (j >= n_orbs && !det.dn.has(j)) {
        det_copy.dn.set(j);
        const size_t det_p1_id = det_p1_to_id[det_copy];
        b_j[det_p1_id] = coef;
      }
    }
    // Solve H^{-1} b_j.
    std::vector<double> G_inv_b_j = green_matrix.solve(b_j);
    for (unsigned i = 0; i < n_orbs * 2; i++) {
      // Construct b_i.
      const auto& b_i = construct_b(i);
      //
      // Calculate <b_i | b_j> and save to res.
      res[i][j] = Util::dot_omp(b_i, G_inv_b_j);
    }
  }

  // Output to file.
  FILE* file = open("green.csv", "w");
  fprintf(file, "i,j,G_ij\n");
  for (unsigned i = 0; i < n_orbs * 2; i++) {
    for (unsigned j = 0; j < n_orbs * 2; j++) {
      fprintf(file, "%u,%u,%.15f\n", i, j, res[i][j]);
    }
  }
}
**/
