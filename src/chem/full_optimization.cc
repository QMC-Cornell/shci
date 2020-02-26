#include "full_optimization.h"

#include "../parallel.h"
#include "../timer.h"
#include <queue>
#include <algorithm>

void FullOptimization::get_orb_param_indices_in_matrix() {
  // return vector of (row,col) indices of optimization parameters
  std::vector<unsigned> orb_sym = integrals_p->orb_sym;
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = i + 1; j < n_orbs; j++) {
      if (orb_sym[i] == orb_sym[j]) {
        orb_param_indices_in_matrix.push_back(std::make_pair(i, j));
      }
    }
  }
  orb_dim = orb_param_indices_in_matrix.size();
  if (Parallel::is_master()) {
    std::cout << "Number of optimization parameters: " << orb_dim
              << std::endl;
  }
}

void FullOptimization::get_indices2index_hashtable() {
  for (size_t i = 0; i < orb_dim; i++) {
    indices2index[orb_param_indices_in_matrix[i]] = i;
  }
}

double FullOptimization::Hessian_ci_orb(const std::vector<Det>& dets, const std::vector<double>& coefs,
    const size_t i_det, const size_t m, const size_t n) {
  double res = 0.;  
  std::vector<unsigned int> orb_sym = integrals_p->orb_sym;
  
  // Create hash table; used for looking up the index of a det
  std::unordered_map<Det, size_t, DetHasher> det2ind;
  for (size_t i = 0; i < dets.size(); i++) {
    det2ind[dets[i]] = i;
  }

  Det this_det = dets[i_det];

  std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
  std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

  // 1 up electrons
  if (this_det.up.has(m)) {
    for (unsigned q = 0; q < n_orbs; q++) {
      if (orb_sym[m] != orb_sym[q]) continue;
      if (m != q && this_det.up.has(q)) continue;

      Det new_det = this_det;
      new_det.up.unset(m);
      new_det.up.set(q);

      double coef;
      if (det2ind.count(new_det) == 1)
        coef = coefs[det2ind[new_det]];
      else
        continue;

      double val = 2 * integrals_p->get_1b(n, q)
          * this_det.up.diff(new_det.up).permutation_factor * coef;
      res += val;
std::cout<<"\n1up <"<<i_det<<"|"<<m<<","<<q<<"|"<<det2ind[new_det]<<"> h_nq "<<n<<" "<<q<<"\t"<<val;
    } 
  }  
  // 1 dn electrons
  if (this_det.dn.has(m)) {
    for (unsigned q = 0; q < n_orbs; q++) {
      if (orb_sym[m] != orb_sym[q]) continue;
      if (m != q && this_det.dn.has(q)) continue;

      Det new_det = this_det;
      new_det.dn.unset(m);
      new_det.dn.set(q);

      double coef;
      if (det2ind.count(new_det) == 1)
        coef = coefs[det2ind[new_det]];
      else
        continue;

      double val = 2 * integrals_p->get_1b(n, q) 
          * this_det.dn.diff(new_det.dn).permutation_factor * coef;
      res += val;
std::cout<<"\n1dn <"<<i_det<<"|"<<m<<","<<q<<"|"<<det2ind[new_det]<<"> h_nq "<<n<<" "<<q<<"\t"<<val;
    }
  } 	

  // 2 up electrons
  if (this_det.up.has(n)) {
    for (unsigned q = 0; q < n_orbs; q++) {
      if (orb_sym[n] != orb_sym[q]) continue;
      if (n != q && this_det.up.has(q)) continue;

      Det new_det = this_det;
      new_det.up.unset(n);
      new_det.up.set(q);

      double coef;
      if (det2ind.count(new_det) == 1)
        coef = coefs[det2ind[new_det]];
      else
        continue;

      double val = -2 * integrals_p->get_1b(m, q) 
          * this_det.up.diff(new_det.up).permutation_factor * coef;
      res += val;
std::cout<<"\n2up <"<<i_det<<"|"<<n<<","<<q<<"|"<<det2ind[new_det]<<"> h_mq "<<m<<" "<<q<<"\t"<<val;          
    } 
  } 
  // 2 dn electrons
  if (this_det.dn.has(n)) {
    for (unsigned q = 0; q < n_orbs; q++) {
      if (orb_sym[n] != orb_sym[q]) continue;
      if (n != q && this_det.dn.has(q)) continue;

      Det new_det = this_det;
      new_det.dn.unset(n);
      new_det.dn.set(q);

      double coef;
      if (det2ind.count(new_det) == 1)
        coef = coefs[det2ind[new_det]];
      else
        continue;
      
      double val = -2 * integrals_p->get_1b(m, q) 
          * this_det.dn.diff(new_det.dn).permutation_factor * coef;
      res += val;
std::cout<<"\n2dn <"<<i_det<<"|"<<n<<","<<q<<"|"<<det2ind[new_det]<<"> h_mq "<<m<<" "<<q<<"\t"<<val;            
    }
  } 
  
  // 3 up electrons
  for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
    unsigned q = occ_up[i_elec];
    if (orb_sym[n] != orb_sym[q]) continue;
    if (n != q && this_det.up.has(n)) continue;

    Det new_det = this_det;
    new_det.up.unset(q);
    new_det.up.set(n);

    double coef;
    if (det2ind.count(new_det) == 1)
      coef = coefs[det2ind[new_det]];
    else
      continue;

    double val = -2 * integrals_p->get_1b(m, q) 
          * new_det.up.diff(this_det.up).permutation_factor * coef;
    res += val;
std::cout<<"\n3up <"<<det2ind[new_det]<<"|"<<n<<","<<q<<"|"<<i_det<<"> h_mq "<<m<<" "<<q<<"\t"<<val;             
  } 
  // 3 dn electrons
  for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
    unsigned q = occ_dn[i_elec];
    if (orb_sym[n] != orb_sym[q]) continue;
    if (n != q && this_det.dn.has(n)) continue;

    Det new_det = this_det;
    new_det.dn.unset(q);
    new_det.dn.set(n);

    double coef;
    if (det2ind.count(new_det) == 1)
      coef = coefs[det2ind[new_det]];
    else
      continue;

    double val = -2 * integrals_p->get_1b(m, q) 
          * new_det.dn.diff(this_det.dn).permutation_factor * coef;
    res += val;
std::cout<<"\n3dn <"<<det2ind[new_det]<<"|"<<n<<","<<q<<"|"<<i_det<<"> h_mq "<<m<<" "<<q<<"\t"<<val;          
  }
  
  // 4 up electrons
  for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
    unsigned q = occ_up[i_elec];
    if (orb_sym[m] != orb_sym[q]) continue;
    if (m != q && this_det.up.has(m)) continue;

    Det new_det = this_det;
    new_det.up.unset(q);
    new_det.up.set(m);

    double coef;
    if (det2ind.count(new_det) == 1)
      coef = coefs[det2ind[new_det]];
    else
      continue;

    double val = 2 * integrals_p->get_1b(n, q) 
          * new_det.up.diff(this_det.up).permutation_factor * coef;
    res += val;
std::cout<<"\n4up <"<<det2ind[new_det]<<"|"<<m<<","<<q<<"|"<<i_det<<"> h_nq "<<n<<" "<<q<<"\t"<<val;  
  } 
  // 4 dn electrons
  for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
    unsigned q = occ_dn[i_elec];
    if (orb_sym[m] != orb_sym[q]) continue;
    if (m != q && this_det.dn.has(m)) continue;

    Det new_det = this_det;
    new_det.dn.unset(q);
    new_det.dn.set(m);

    double coef;
    if (det2ind.count(new_det) == 1)
      coef = coefs[det2ind[new_det]];
    else
      continue;

    double val = 2 * integrals_p->get_1b(n, q) 
          * new_det.dn.diff(this_det.dn).permutation_factor * coef;
    res += val;
std::cout<<"\n4dn <"<<det2ind[new_det]<<"|"<<m<<","<<q<<"|"<<i_det<<"> h_nq "<<n<<" "<<q<<"\t"<<val;  
  }
  
  //////////////////////// 2b ////////////////////////////
  // 1 up electrons
  
  
  return res;
}

void FullOptimization::get_1rdm(
    const std::vector<Det>& dets, const std::vector<double>& coefs, const bool dump_csv) {
  //=====================================================
  // Create 1RDM using the variational wavefunction
  //
  // Created: Y. Yao, June 2018
  //=====================================================
  bool time_sym = Config::get<bool>("time_sym", false);

  std::vector<unsigned int> orb_sym = integrals_p->orb_sym;

  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the index of a det
  std::unordered_map<Det, size_t, DetHasher> det2ind;
  for (size_t i = 0; i < dets.size(); i++) {
    det2ind[dets[i]] = i;
  }

#pragma omp parallel for
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    Det this_det = dets[i_det];

    std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
    std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

    // up electrons
    for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
      unsigned p = occ_up[i_elec];
      for (unsigned r = 0; r < n_orbs; r++) {
        if (orb_sym[p] != orb_sym[r]) continue;
        if (p != r && this_det.up.has(r)) continue;

        Det new_det = this_det;
        new_det.up.unset(p);
        new_det.up.set(r);

        double coef;
        if (det2ind.count(new_det) == 1)
          coef = coefs[det2ind[new_det]];
        else
          continue;
          
        write_in_1rdm_and_hessian(p, r, this_det.up.diff(new_det.up).permutation_factor,
		       i_det, coefs[i_det], det2ind[new_det], coef);
      }  // r
    }  // i_elec

    // dn electrons
    if (!time_sym) {  // If time_sym, then the up excitations will equal to the dn excitations
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        for (unsigned r = 0; r < n_orbs; r++) {
          if (orb_sym[p] != orb_sym[r]) continue;
          if (p != r && this_det.dn.has(r)) continue;

          Det new_det = this_det;
          new_det.dn.unset(p);
          new_det.dn.set(r);

          double coef;
          if (det2ind.count(new_det) == 1)
            coef = coefs[det2ind[new_det]];
          else
            continue;

          write_in_1rdm_and_hessian(p, r, this_det.dn.diff(new_det.dn).permutation_factor, 
			  i_det, coefs[i_det],  det2ind[new_det], coef);
        }  // r
      }  // i_elec
    }
  }  // i_det

  if (time_sym) {
    one_rdm *= 2.;
    hessian_ci_orb *= 2.;
  }

  if (Parallel::is_master()) {
    if (dump_csv) {
      FILE* pFile;
      pFile = fopen("1rdm.csv", "w");
      fprintf(pFile, "p,r,1rdm\n");
      for (unsigned p = 0; p < n_orbs; p++) {
        for (unsigned r = p; r < n_orbs; r++) {
          const double rdm_pr = one_rdm(p, r);
          if (std::abs(rdm_pr) < 1e-9) continue;
          fprintf(
              pFile,
              "%d,%d,%#.15g\n",
              integrals_p->orb_order[p],
              integrals_p->orb_order[r],
              rdm_pr);
        }
      }
      fclose(pFile);
    }
  }
  Timer::checkpoint("computing 1RDM");
}

void FullOptimization::write_in_1rdm_and_hessian(size_t p, size_t q, double perm_fac, 
		size_t i_det, double coef_i, size_t j_det, double coef_j) {
  // <c^dag_p c_q>
  // i is the CI index
#pragma omp atomic
  one_rdm(p, q) += perm_fac * coef_i * coef_j;
  
  if (i_det < n_dets_truncate) {
    for (unsigned t = p + 1; t < n_orbs; t++) {
      if (indices2index.count(std::make_pair(p, t))==1) {
#pragma omp atomic
        hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += 2 * integrals_p->get_1b(t, q) * perm_fac * coef_j;
      }
    }
    for (unsigned t = 0; t < p; t++) {
      if (indices2index.count(std::make_pair(t, p))==1) {
#pragma omp atomic
        hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, p))) -= 2 * integrals_p->get_1b(t, q) * perm_fac * coef_j;
      }
    }
  }
  if (j_det < n_dets_truncate) {
    for (unsigned t = 0; t < p; t++) {
      if (indices2index.count(std::make_pair(t, p))==1) {
#pragma omp atomic
        hessian_ci_orb(j_det, indices2index.at(std::make_pair(t, p))) -= 2 * integrals_p->get_1b(t, q) * perm_fac * coef_i;
      }
    }
    for (unsigned t = p + 1; t < n_orbs; t++) {
      if (indices2index.count(std::make_pair(p, t))==1) {
#pragma omp atomic
        hessian_ci_orb(j_det, indices2index.at(std::make_pair(p, t))) += 2 * integrals_p->get_1b(t, q) * perm_fac * coef_i;
      }
    }
  }
}

inline size_t FullOptimization::combine4_2rdm(size_t p, size_t q, size_t r, size_t s) const {
  size_t a = p * n_orbs + s;
  size_t b = q * n_orbs + r;
  if (a > b) {
    return (a * (a + 1)) / 2 + b;
  } else {
    return (b * (b + 1)) / 2 + a;
  }
}

int FullOptimization::permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const {
  // Calculate the permutation factor of
  // c^dag_p c^dag_q c_r c_s |halfket>

  unsigned counter = 0;

  // annihilation operators
  std::vector<unsigned> orbs_a{s, r};
  for (unsigned iorb = 0; iorb < 2; iorb++) {
    unsigned orb = orbs_a[iorb];
    counter += halfket.bit_till(orb);
    halfket.unset(orb);
  }

  // creation operators
  std::vector<unsigned> orbs_c{q, p};
  for (unsigned iorb = 0; iorb < 2; iorb++) {
    unsigned orb = orbs_c[iorb];
    counter += halfket.bit_till(orb);
    halfket.set(orb);
  }

  if (counter % 2 == 0)
    return 1;
  else
    return -1;
}


void FullOptimization::get_2rdm(
    const std::vector<Det>& dets,
    const std::vector<double>& coefs,
    const std::vector<std::vector<size_t>>& connections) {
  //=====================================================
  // Create spatial 2RDM using the variational wavefunction
  // and Hamiltonian connections.
  //
  // D_pqrs corresponds to <a^+_p a^+_q a_r a_s>
  // Four different spin configurations
  // (1) p: up, q: up, r: up, s: up
  // (2) p: dn, q: dn, r: dn, s: dn
  // (3) p: dn, q: up, r: up, s: dn
  // (4) p: up, q: dn, r: dn, s: up
  //
  // Symmetry (p,s)<->(q,r) is used to reduce storage by half.
  //
  // When time_sym is used, the input wavefunction and Hamiltonian
  // connections should NOT be unpacked.
  //
  // Created: Y. Yao, August 2018
  // Modified: Y. Yao, October 2018: MPI compatibility
  //=====================================================
  bool time_sym = Config::get<bool>("time_sym", false);

  size_t size_two_rdm = n_orbs * n_orbs * (n_orbs * n_orbs + 1) / 2;
  two_rdm.resize(size_two_rdm, 0.);

#pragma omp parallel for schedule(dynamic, 5)
  for (size_t i_det = 0; i_det < connections.size(); i_det++) {
    Det this_det = dets[i_det];
    double this_coef = coefs[i_det];

    for (size_t j_det = 0; j_det < connections[i_det].size(); j_det++) {
      size_t connected_ind = connections[i_det][j_det];
      Det connected_det = dets[connected_ind];
      double connected_coef = coefs[connected_ind];

      if (!time_sym)
        get_2rdm_elements(connected_det, connected_coef, connected_ind, this_det, this_coef, i_det);
      else {
        if (this_det.up == this_det.dn) {
          if (connected_det.up == connected_det.dn) {
            get_2rdm_elements(connected_det, connected_coef, connected_ind, this_det, this_coef, i_det);
          } else {
            Det connected_det_rev = connected_det;
            connected_det_rev.reverse_spin();
            double connected_coef_new = connected_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef_new, connected_ind, this_det, this_coef, i_det);
            get_2rdm_elements(connected_det_rev, connected_coef_new, connected_ind, this_det, this_coef, i_det);
          }
        } else {
          if (connected_det.up == connected_det.dn) {
            Det this_det_rev = this_det;
            this_det_rev.reverse_spin();
            double this_coef_new = this_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef, connected_ind, this_det, this_coef_new, i_det);
            get_2rdm_elements(connected_det, connected_coef, connected_ind, this_det_rev, this_coef_new, i_det);
          } else {
            Det connected_det_rev = connected_det;
            connected_det_rev.reverse_spin();
            double connected_coef_new = connected_coef * Util::SQRT2_INV;
            Det this_det_rev = this_det;
            this_det_rev.reverse_spin();
            double this_coef_new = this_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef_new, connected_ind, this_det, this_coef_new, i_det);
            if (j_det != 0)
              get_2rdm_elements(connected_det, connected_coef_new, connected_ind, this_det_rev, this_coef_new, i_det);
            get_2rdm_elements(connected_det_rev, connected_coef_new, connected_ind, this_det, this_coef_new, i_det);
            get_2rdm_elements(connected_det_rev, connected_coef_new, connected_ind, this_det_rev, this_coef_new, i_det);
          }
        }
      }
    }
  }

  if (Parallel::get_n_procs() > 1) {
    std::vector<double> global_two_rdm(size_two_rdm);

    double* src_ptr = two_rdm.data();
    double* dest_ptr = global_two_rdm.data();

    const size_t CHUNK_SIZE = 1 << 27;
    unsigned n_elems_left = size_two_rdm;

    while (n_elems_left > CHUNK_SIZE) {
      //MPI_Reduce(src_ptr, dest_ptr, CHUNK_SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Allreduce(src_ptr, dest_ptr, CHUNK_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      n_elems_left -= CHUNK_SIZE;
      src_ptr += CHUNK_SIZE;
      dest_ptr += CHUNK_SIZE;
    }
    //MPI_Reduce(src_ptr, dest_ptr, n_elems_left, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(src_ptr, dest_ptr, n_elems_left, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    two_rdm = global_two_rdm;
  }

  Timer::checkpoint("computing 2RDM");
}

void FullOptimization::get_2rdm_elements(
    const Det& connected_det,
    const double connected_coef,
    const size_t j_det,
    const Det& this_det,
    const double this_coef,
    const size_t i_det) {
  //=====================================================
  // Fill in 2RDM for a given pair of dets. When the two dets
  // are not idential also do the pair in reverse order since
  // the Hamiltonian is upper triangular.
  //
  // D_pqrs corresponds to <a^+_p a^+_q a_r a_s>
  // Four different spin configurations
  // (1) p: up, q: up, r: up, s: up
  // (2) p: dn, q: dn, r: dn, s: dn
  // (3) p: dn, q: up, r: up, s: dn
  // (4) p: up, q: dn, r: dn, s: up
  //
  // Created: Y. Yao, August 2018
  //=====================================================

  std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
  std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

  // 0 alpha excitation apart
  if (this_det.up == connected_det.up) {
    if (this_det.dn == connected_det.dn) {  // 0 beta excitation apart

      // (1)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        for (unsigned j_elec = i_elec + 1; j_elec < n_up; j_elec++) {
          unsigned s = occ_up[i_elec];
          unsigned r = occ_up[j_elec];

          write_in_2rdm_and_hessian(s, r, r, s, 1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(s, r, s, r, -1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(r, s, s, r, 1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(r, s, r, s, -1., i_det, this_coef, j_det, connected_coef);
        }
      }

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        for (unsigned j_elec = i_elec + 1; j_elec < n_dn; j_elec++) {
          unsigned s = occ_dn[i_elec];
          unsigned r = occ_dn[j_elec];

          write_in_2rdm_and_hessian(s, r, r, s, 1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(s, r, s, r, -1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(r, s, s, r, 1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(r, s, r, s, -1., i_det, this_coef, j_det, connected_coef);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        for (unsigned j_elec = 0; j_elec < n_dn; j_elec++) {
          unsigned s = occ_up[i_elec];
          unsigned r = occ_dn[j_elec];

          write_in_2rdm_and_hessian(s, r, r, s, 1., i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(r, s, s, r, 1., i_det, this_coef, j_det, connected_coef);
        }
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // to

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        if ((p != b1) && (p != b2)) {
          double perm_fac = permfac_ccaa(this_det.dn, p, b2, b1, p);

          write_in_2rdm_and_hessian(p, b2, b1, p, perm_fac, i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(b2, p, b1, p, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(p, b2, p, b1, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(b2, p, p, b1, perm_fac, i_det, this_coef, j_det, connected_coef);

          perm_fac = permfac_ccaa(connected_det.dn, p, b1, b2, p);

          write_in_2rdm_and_hessian(p, b1, b2, p, perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(b1, p, b2, p, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(p, b1, p, b2, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(b1, p, p, b2, perm_fac, i_det, this_coef, j_det, connected_coef);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up[i_elec];
        double perm_fac = connected_det.dn.diff(this_det.dn).permutation_factor;

        write_in_2rdm_and_hessian(p, b2, b1, p, perm_fac, i_det, this_coef, j_det, connected_coef);
        write_in_2rdm_and_hessian(b2, p, p, b1, perm_fac,i_det, this_coef, j_det, connected_coef);

        perm_fac = this_det.dn.diff(connected_det.dn).permutation_factor;

        write_in_2rdm_and_hessian(p, b1, b2, p, perm_fac,i_det, this_coef, j_det, connected_coef);
        write_in_2rdm_and_hessian(b1, p, p, b2, perm_fac,i_det, this_coef, j_det, connected_coef);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 2) {  // 2 beta excitations apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).right_only[1];
      unsigned b3 = connected_det.dn.diff(this_det.dn).left_only[0];  // to
      unsigned b4 = connected_det.dn.diff(this_det.dn).left_only[1];

      double perm_fac = connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm_and_hessian(b3, b4, b2, b1, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b3, b4, b1, b2, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b4, b3, b2, b1, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b4, b3, b1, b2, perm_fac, i_det, this_coef, j_det, connected_coef);

      perm_fac = this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm_and_hessian(b1, b2, b4, b3, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b1, b2, b3, b4, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b2, b1, b4, b3, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b2, b1, b3, b4, perm_fac, i_det, this_coef, j_det, connected_coef);
    }

    // 1 alpha excitation apart
  } else if (connected_det.up.diff(this_det.up).n_diffs == 1) {
    if (this_det.dn == connected_det.dn) {  // 0 beta excitation apart
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // from
      unsigned a2 = connected_det.up.diff(this_det.up).left_only[0];  // to

      // (1)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up[i_elec];
        if ((p != a1) && (p != a2)) {
          double perm_fac = permfac_ccaa(this_det.up, p, a2, a1, p);

          write_in_2rdm_and_hessian(p, a2, a1, p, perm_fac,i_det, this_coef, j_det, connected_coef);  
          write_in_2rdm_and_hessian(p, a2, p, a1, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(a2, p, a1, p, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(a2, p, p, a1, perm_fac, i_det, this_coef, j_det, connected_coef);

          perm_fac = permfac_ccaa(connected_det.up, p, a1, a2, p);

          write_in_2rdm_and_hessian(p, a1, a2, p, perm_fac, i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(p, a1, p, a2, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(a1, p, a2, p, -perm_fac,i_det, this_coef, j_det, connected_coef);
          write_in_2rdm_and_hessian(a1, p, p, a2, perm_fac, i_det, this_coef, j_det, connected_coef);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];

        double perm_fac = connected_det.up.diff(this_det.up).permutation_factor;

        write_in_2rdm_and_hessian(a2, p, p, a1, perm_fac, i_det, this_coef, j_det, connected_coef);
        write_in_2rdm_and_hessian(p, a2, a1, p, perm_fac, i_det, this_coef, j_det, connected_coef);

        perm_fac = this_det.up.diff(connected_det.up).permutation_factor;

        write_in_2rdm_and_hessian(a1, p, p, a2, perm_fac, i_det, this_coef, j_det, connected_coef);
        write_in_2rdm_and_hessian(p, a1, a2, p, perm_fac, i_det, this_coef, j_det, connected_coef);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // up from
      unsigned a2 = connected_det.up.diff(this_det.up).left_only[0];  // up to
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // dn from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // dn to

      double perm_fac = connected_det.up.diff(this_det.up).permutation_factor *
                       connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm_and_hessian(a2, b2, b1, a1, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b2, a2, a1, b1, perm_fac, i_det, this_coef, j_det, connected_coef);

      perm_fac = this_det.up.diff(connected_det.up).permutation_factor *
                this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm_and_hessian(a1, b1, b2, a2, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(b1, a1, a2, b2, perm_fac, i_det, this_coef, j_det, connected_coef);
    }

    // 2 alpha excitations apart
  } else if (connected_det.up.diff(this_det.up).n_diffs == 2) {
    if (this_det.dn == connected_det.dn) {
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // from
      unsigned a2 = connected_det.up.diff(this_det.up).right_only[1];
      unsigned a3 = connected_det.up.diff(this_det.up).left_only[0];  // to
      unsigned a4 = connected_det.up.diff(this_det.up).left_only[1];

      double perm_fac = connected_det.up.diff(this_det.up).permutation_factor;

      write_in_2rdm_and_hessian(a3, a4, a2, a1, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a3, a4, a1, a2, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a4, a3, a2, a1, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a4, a3, a1, a2, perm_fac, i_det, this_coef, j_det, connected_coef);

      perm_fac = this_det.up.diff(connected_det.up).permutation_factor;

      write_in_2rdm_and_hessian(a1, a2, a4, a3, perm_fac, i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a1, a2, a3, a4, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a2, a1, a4, a3, -perm_fac,i_det, this_coef, j_det, connected_coef);
      write_in_2rdm_and_hessian(a2, a1, a3, a4, perm_fac, i_det, this_coef, j_det, connected_coef);
    }
  }
}

void FullOptimization::write_in_2rdm_and_hessian(size_t p, size_t q, size_t r, size_t s, double perm_fac,
		size_t i_det, double coef_i, size_t j_det, double coef_j) {
  // By symmetry (p,s)<->(q,r) only half of the 2RDM needs storing.
  size_t a = p * n_orbs + s;
  size_t b = q * n_orbs + r;
  if (a >= b)
#pragma omp atomic
    two_rdm[(a * (a + 1)) / 2 + b] += perm_fac * coef_i * coef_j;
    

  if (i_det < n_dets_truncate) {
    for (unsigned t = p + 1; t < n_orbs; t++) {
      if (indices2index.count(std::make_pair(p, t))==1) {
#pragma omp atomic
        hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += 2 * coef_j * integrals_p->get_2b(t, s, q, r) * perm_fac;
      }
    }
    for (unsigned t = 0; t < p; t++) {
      if (indices2index.count(std::make_pair(t, p))==1) {
#pragma omp atomic
        hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, p))) -= 2 * coef_j * integrals_p->get_2b(t, s, q, r) * perm_fac;
      }
    }
  }
  if (j_det < n_dets_truncate) {
    for (unsigned t = 0; t < p; t++) {
      if (indices2index.count(std::make_pair(t, p))==1) {
#pragma omp atomic
        hessian_ci_orb(j_det, indices2index.at(std::make_pair(t, p))) -= 2 * coef_i * integrals_p->get_2b(t, s, q, r) * perm_fac;
      }
    }
    for (unsigned t = p + 1; t < n_orbs; t++) {
      if (indices2index.count(std::make_pair(p, t))==1) {
#pragma omp atomic
        hessian_ci_orb(j_det, indices2index.at(std::make_pair(p, t))) += 2 * coef_i * integrals_p->get_2b(t, s, q, r) * perm_fac;
      }
    }
  }
}

double FullOptimization::one_rdm_elem(unsigned p, unsigned q) const { return one_rdm(p, q); }

double FullOptimization::two_rdm_elem(unsigned p, unsigned q, unsigned r, unsigned s) const {
  return two_rdm[combine4_2rdm(p, q, r, s)];
}


void FullOptimization::rotate_integrals() {
  new_integrals.resize(n_orbs);
  Integrals_array tmp_integrals(n_orbs);

#pragma omp parallel for
  for (unsigned i = 0; i < n_orbs; i++) {
    new_integrals[i].resize(n_orbs);
    tmp_integrals[i].resize(n_orbs);
    for (unsigned j = 0; j < n_orbs; j++) {
      new_integrals[i][j].resize(n_orbs + 1);
      tmp_integrals[i][j].resize(n_orbs + 1);

      for (unsigned k = 0; k < n_orbs + 1; k++) {
        new_integrals[i][j][k].resize(n_orbs + 1);
        tmp_integrals[i][j][k].resize(n_orbs + 1);
        std::fill(new_integrals[i][j][k].begin(), new_integrals[i][j][k].end(),
                  0.);
        std::fill(tmp_integrals[i][j][k].begin(), tmp_integrals[i][j][k].end(),
                  0.);
      }
    }
  }

// Two-body integrals
#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          tmp_integrals[p][q][r][s] = integrals_p->get_2b(p, q, r, s);
        } // s
      }   // r
    }     // q
  }       // p

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, p) * tmp_integrals[i][q][r][s];
          }
          new_integrals[p][q][r][s] = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, q) * tmp_integrals[p][i][r][s];
          }
          new_integrals[p][q][r][s] = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, r) * tmp_integrals[p][q][i][s];
          }
          new_integrals[p][q][r][s] = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, s) * tmp_integrals[p][q][r][i];
          }
          new_integrals[p][q][r][s] = new_val;
        } // s
      }   // r
    }     // q
  }       // p

// One-body integrals
#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      tmp_integrals[p][q][n_orbs][n_orbs] = integrals_p->get_1b(p, q);
    }
  }

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i = 0; i < n_orbs; i++) {
        new_val += rot(i, p) * tmp_integrals[i][q][n_orbs][n_orbs];
      }
      new_integrals[p][q][n_orbs][n_orbs] = new_val;
    }
  }

  tmp_integrals = new_integrals;

#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i = 0; i < n_orbs; i++) {
        new_val += rot(i, q) * tmp_integrals[p][i][n_orbs][n_orbs];
      }
      new_integrals[p][q][n_orbs][n_orbs] = new_val;
    }
  }
  Timer::checkpoint("rotate integrals");
}

void FullOptimization::rewrite_integrals() {
  // replace integrals with new_integrals
  integrals_p->integrals_2b.clear();
  integrals_p->integrals_1b.clear();

  unsigned p, q, r, s;
  double value;
  for (p = 0; p < n_orbs; p++) {
    for (q = 0; q < n_orbs; q++) {
      for (r = 0; r < n_orbs; r++) {
        for (s = 0; s < n_orbs; s++) {
          value = new_integrals[p][q][r][s];
          integrals_p->integrals_2b.set(Integrals::combine4(p, q, r, s), value,
                                        [&](double &a, const double &b) {
                                          if (std::abs(a) < std::abs(b))
                                            a = b;
                                        });
        }
      }
    }
  }

  for (p = 0; p < n_orbs; p++) {
    for (q = 0; q < n_orbs; q++) {
      value = new_integrals[p][q][n_orbs][n_orbs];
      integrals_p->integrals_1b.set(Integrals::combine2(p, q), value,
                                    [&](double &a, const double &b) {
                                      if (std::abs(a) < std::abs(b))
                                        a = b;
                                    });
    }
  }
}

std::vector<size_t> FullOptimization::get_most_important_indices(
        const VectorXd& gradient,
        const MatrixXd& hessian,
        const double quantile) const {
  std::priority_queue<std::pair<double, size_t>> q;
  std::cout<<"\ng^2/2h ";
  for (size_t i=0; i<orb_dim; i++) {
    double val = std::abs(std::pow(gradient(i),2) / 2. / hessian(i,i));
    std::cout<<val<<" ";
    q.push(std::pair<double, size_t>(val, i));
  }
  size_t num_nonzero = orb_dim * quantile;
  std::cout <<"\nnum_nonzero = "<<num_nonzero << std::endl;
  std::vector<size_t> res;
  for (size_t i=0; i<num_nonzero; i++) {
    res.push_back(q.top().second);
    q.pop();
  }
  return res;
}

void FullOptimization::generate_optorb_integrals_from_newton(
	const std::vector<double>& row_sum,
	const std::vector<double>& diag,
	const std::vector<double>& coefs,
	const double E_var) {
  VectorXd grad = gradient(orb_param_indices_in_matrix);
  MatrixXd hess = hessian(orb_param_indices_in_matrix);
  //MatrixXd hess = hessian_diagonal(orb_param_indices_in_matrix).asDiagonal();
  one_rdm.resize(0, 0);
  two_rdm.clear(); two_rdm.shrink_to_fit();
 
  // one more contribution to hessian_co in addition to rdm elements
#pragma omp parallel for
  for (size_t i = 0; i < n_dets_truncate; i++) {
    hessian_ci_orb.row(i) -= 2. * coefs[i] * grad.transpose();
    //app_hessian_ci_orb.row(i) -= 2. * coefs[i] * grad.transpose();
  }

//std::cout<<"\nHoo\n"<<hess;
//std::cout<<"\nHco\n"<<hessian_ci_orb.topRows(20);
//std::cout<<"\nHcc\n"<<hessian_ci_ci.topRows(20);
//for (size_t i=0; i<n_dets_truncate/500; i++) {
//std::cout<<"\ni="<<i*500<<"\t"<<hessian_ci_orb.row(i*500);
//} 
  /*
  MatrixXd BTAinv(orb_dim, n_dets_truncate);
  for (size_t i = 0; i < n_dets_truncate; i++) {
    double Hcc_diag = -8. * coefs[i] * row_sum[i] + 2. * diag[i] + (8. * coefs[i] * coefs[i] - 2.) * E_var;
    BTAinv.col(i) = hessian_ci_orb.row(i).transpose() / Hcc_diag;
  }
  MatrixXd BTAinvB = BTAinv * hessian_ci_orb;
  VectorXd tmp = VectorXd::Zero(orb_dim); // B^T*A^inv*g
  for (size_t i = 0; i < n_dets_truncate; i++) {
    tmp += 2. * (row_sum[i] - coefs[i] * E_var) * BTAinv.col(i);
  }
  // rotation matrix
  //VectorXd new_param = (hess-BTAinvB).householderQr().solve(tmp - grad);
  */

  std::vector<size_t> nonzero_indices = get_most_important_indices(grad, hess, Config::get<double>("optimization/param_proportion", 1.));
  for (size_t i=0; i<orb_dim; i++) {
    if (std::find(nonzero_indices.begin(), nonzero_indices.end(), i) != nonzero_indices.end()) continue;
    grad(i) = 0.;
    for (size_t j=0; j<n_dets_truncate; j++) {
      hessian_ci_orb(j,i) = 0.;
    }
    for (size_t j=0; j<orb_dim; j++) {
      hess(i,j) = 0.;
      hess(j,i) = 0.;
    }
  }

  VectorXd grad_c(n_dets_truncate);
#pragma omp parallel for
  for (size_t i = 0; i < n_dets_truncate; i++) grad_c(i) = 2. * (row_sum[i] - coefs[i] * E_var);
  MatrixXd tmp = hessian_ci_orb.bottomRows(n_dets_truncate-1);
  tmp = hessian_ci_ci.bottomRightCorner(n_dets_truncate-1, n_dets_truncate-1).householderQr().solve(tmp);
  hess -= hessian_ci_orb.bottomRows(n_dets_truncate-1).transpose() * tmp;
  // rotation matrix
  VectorXd new_param = hess.fullPivLu().solve(- grad);
 
  if (Parallel::is_master())
    printf("norm of gradient: %.5f, norm of update: %.5f.\n", grad.norm(),
           new_param.norm());

  fill_rot_matrix_with_parameters(new_param, orb_param_indices_in_matrix);
  rotate_integrals();
}

void FullOptimization::get_approximate_hessian_ci_orb(const SparseMatrix& hamiltonian_matrix, 
                                                      const double E_var,
                                                      const std::vector<Det>& dets,
                                                      const std::vector<double>& coefs) {
  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
  }

  app_hessian_ci_orb = MatrixXd::Zero(n_dets_truncate, orb_dim);
  
  std::vector<unsigned int> orb_sym = integrals_p->orb_sym;
//#pragma omp parallel for
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    //Det this_det = dets[i_det];

    //for (size_t j_id = 0; j_id < hamiltonian_matrix.rows[i_det].size(); j_id++) {
    for (size_t j_id = 1; j_id < hamiltonian_matrix.rows[i_det].size(); j_id++) {
      size_t j_det = hamiltonian_matrix.rows[i_det].get_index(j_id);
      Det this_det = dets[j_det];
      std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
      std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();
      // up
      for (const unsigned t: occ_up) {
        for (unsigned q = t + 1; q < n_orbs; q++) {
          if (orb_sym[t] != orb_sym[q]) continue;
          if (this_det.up.has(q)) continue;
          Det new_det = this_det;
          new_det.up.unset(t); new_det.up.set(q);
          double coef;
          if (det2coef.count(new_det) == 1)
            coef = det2coef[new_det];
          else
            continue;
          int permfac = this_det.up.diff(new_det.up).permutation_factor;
          app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, q))) -= 2. * hamiltonian_matrix.rows[i_det].get_value(j_id) * coef * permfac;
          if (j_id == 0) app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, q))) -= - 2. * E_var * coef * permfac;
        }
        for (unsigned p = 0; p < t; p++) {
          if (orb_sym[t] != orb_sym[p]) continue;
          if (this_det.up.has(p)) continue;
          Det new_det = this_det;
          new_det.up.unset(t); new_det.up.set(p);
          double coef;
          if (det2coef.count(new_det) == 1)
            coef = det2coef[new_det];
          else
            continue;
          int permfac = this_det.up.diff(new_det.up).permutation_factor;
          app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += 2. * hamiltonian_matrix.rows[i_det].get_value(j_id) * coef * permfac;
          if (j_id == 0) app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += - 2. * E_var * coef * permfac;
        }
      }

      // dn 
      for (const unsigned t: occ_dn) {
        for (unsigned q = t + 1; q < n_orbs; q++) {
          if (orb_sym[t] != orb_sym[q]) continue;
          if (this_det.dn.has(q)) continue;
          Det new_det = this_det;
          new_det.dn.unset(t); new_det.dn.set(q);
          double coef;
          if (det2coef.count(new_det) == 1)
            coef = det2coef[new_det];
          else
            continue;
          int permfac = this_det.dn.diff(new_det.dn).permutation_factor;
          app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, q))) -= 2. * hamiltonian_matrix.rows[i_det].get_value(j_id) * coef * permfac;
          if (j_id == 0) app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(t, q))) -= - 2. * E_var * coef * permfac;
        }
        for (unsigned p = 0; p < t; p++) {
          if (orb_sym[t] != orb_sym[p]) continue;
          if (this_det.dn.has(p)) continue;
          Det new_det = this_det;
          new_det.dn.unset(t); new_det.dn.set(p);
          double coef;
          if (det2coef.count(new_det) == 1)
            coef = det2coef[new_det];
          else
            continue;
          int permfac = this_det.dn.diff(new_det.dn).permutation_factor;
          app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += 2. * hamiltonian_matrix.rows[i_det].get_value(j_id) * coef * permfac;
          if (j_id == 0) app_hessian_ci_orb(i_det, indices2index.at(std::make_pair(p, t))) += - 2. * E_var * coef * permfac;
        }
      }
    }
  }
  Timer::checkpoint("computing approximate_hessian_ci_orb");
}

void FullOptimization::get_hessian_ci_ci(const SparseMatrix& hamiltonian_matrix, 
		const std::vector<double>& row_sum,
		const std::vector<double>& coefs,
		const double E_var) {
  hessian_ci_ci = MatrixXd::Zero(n_dets_truncate, n_dets_truncate);
#pragma omp parallel for
  for (size_t i = 0; i < n_dets_truncate; i++) {
    for (size_t j_id = 0; j_id < hamiltonian_matrix.rows[i].size(); j_id++) {
      size_t j = hamiltonian_matrix.rows[i].get_index(j_id);
      if (j >= n_dets_truncate) continue;
      double val = hamiltonian_matrix.rows[i].get_value(j_id);
      hessian_ci_ci(i, j) = 2. * val;
      if (j_id != 0) hessian_ci_ci(j, i) = 2. * val;
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < n_dets_truncate; i++) { //TODO: improve with symmetry
    hessian_ci_ci(i, i) -= 2. * E_var;
    for (size_t j = 0; j < n_dets_truncate; j++) {
      hessian_ci_ci(i, j) += 8. * coefs[i] * coefs[j] * E_var - 4. * coefs[i] * row_sum[j] - 4. * coefs[j] * row_sum[i];
    }
  }
  Timer::checkpoint("computing hessian_ci_ci");
}

void FullOptimization::get_approximate_hessian_ci_ci(const SparseMatrix& hamiltonian_matrix, 
		const std::vector<double>& row_sum,
		const std::vector<double>& coefs,
		const double E_var) {
  hessian_ci_ci = MatrixXd::Zero(n_dets_truncate, n_dets_truncate);
  size_t N = coefs.size();
//diagonal
#pragma omp parallel for
  for (size_t i = 0; i < n_dets_truncate; i++) {
    hessian_ci_ci(i, i) += 2. * hamiltonian_matrix.rows[i].get_value(0) + (8. * coefs[i] * coefs[i] - 2.) * E_var - 8. * coefs[i] * row_sum[i];
  }
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    for (size_t j_id = 1; j_id < hamiltonian_matrix.rows[i].size(); j_id++) {
      size_t j = hamiltonian_matrix.rows[i].get_index(j_id);
      assert( j != i);
      double val = hamiltonian_matrix.rows[i].get_value(j_id);
      hessian_ci_ci(i, j) += 2. * val;
      hessian_ci_ci(j, i) += 2. * val;
    }
  }
  Timer::checkpoint("computing approximate hessian_ci_ci");
}

/*
void FullOptimization::generate_optorb_integrals_from_approximate_newton(const std::vector<double>& row_sum) {
  VectorXd grad = gradient(orb_param_indices_in_matrix);
  MatrixXd hess_diag = hessian_diagonal(orb_param_indices_in_matrix);
  one_rdm.resize(0, 0);
  two_rdm.clear(); two_rdm.shrink_to_fit();

  VectorXd new_param(orb_dim);
  for (size_t i = 0; i < orb_dim; i++) {
    hess_diag(i) = (hess_diag(i) > 0 ? std::max(hess_diag(i), 1e-5)
                                     : std::min(hess_diag(i), -1e-5));
    new_param(i) = -grad(i) / hess_diag(i);
  }

  static double eps = 0.05;
  static bool is_first_iter = true;
  static VectorXd old_param(dim), old_update(dim);
  VectorXd new_update;
  if (Config::get<bool>("optimization/accelerate", false)) {
    if (Parallel::is_master())
      printf("Accelerate optimization by overshooting.\n");
    if (is_first_iter) {
      new_update = 4 * new_param;
      old_param = new_param;
      old_update = new_update;
      is_first_iter = false;
    } else {
      new_update = new_param;
      old_param = new_param;

      double new_norm = 0., old_norm = 0., inner_prod = 0.;

      for (size_t i = 0; i < dim; i++) {
        inner_prod += old_update(i) * new_update(i) * hess_diag(i);
        new_norm += new_update(i) * new_update(i) * hess_diag(i);
        old_norm += old_update(i) * old_update(i) * hess_diag(i);
      }
      new_norm = std::sqrt(new_norm);
      old_norm = std::sqrt(old_norm);
      double cos = inner_prod / new_norm / old_norm;

      double step_size = std::max(std::min(2. / (1. - cos), 1 / eps), 1.);
      if (inner_prod < 0.) {
        eps = std::min(std::pow(eps, .8), .5);
        if (Parallel::is_master())
          printf("eps for Newton step enhancement changes to %.3f.\n", eps);
      }
      new_update *= step_size;
      old_update = new_update;
      if (Parallel::is_master())
        printf("cosine: %.5f, step size: %.5f.\n", cos, step_size);
    }
  } else {
    new_update = std::move(new_param);
  }
  if (Parallel::is_master())
    printf("norm of gradient: %.5f, norm of update: %.5f.\n", grad.norm(),
           new_update.norm());

  fill_rot_matrix_with_parameters(new_update, param_indices);
  rotate_integrals();
}
*/

/*
std::vector<FullOptimization::index_t> FullOptimization::parameter_indices() const {
  // return vector of (row,col) indices of optimization parameters
  std::vector<unsigned> orb_sym = integrals_p->orb_sym;
  std::vector<index_t> indices;
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = i + 1; j < n_orbs; j++) {
      if (orb_sym[i] == orb_sym[j]) {
        indices.push_back(std::make_pair(i, j));
      }
    }
  }
  if (Parallel::is_master()) {
    std::cout << "Number of optimization parameters: " << indices.size()
              << std::endl;
  }
  return indices;
}*/

void FullOptimization::fill_rot_matrix_with_parameters(
    const VectorXd &parameters, const std::vector<index_t> &parameter_indices) {
  // fill in the rot matrix with parameters, with entry indices being
  // parameter_indices
  MatrixXd X_matrix = MatrixXd::Zero(n_orbs, n_orbs);
  for (unsigned i = 0; i < parameter_indices.size(); i++) {
    unsigned p = parameter_indices[i].first;
    unsigned q = parameter_indices[i].second;
    X_matrix(p, q) = -1 * parameters(i);
    X_matrix(q, p) = parameters(i);
  }

  SelfAdjointEigenSolver<MatrixXd> es(n_orbs);
  es.compute(X_matrix * X_matrix);
  MatrixXd Tau2, W_matrix;
  Tau2 = es.eigenvalues().transpose(); // Tau^2
  W_matrix = es.eigenvectors();

  double tau, cos_tau, sinc_tau;
#pragma omp parallel for
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = 0; j < n_orbs; j++) {
      for (unsigned k = 0; k < n_orbs; k++) {
        if (std::abs(Tau2(k)) < 1e-10) {
          cos_tau = 1;
          sinc_tau = 1;
        } else {
          tau = std::sqrt(-Tau2(k));
          cos_tau = std::cos(tau);
          sinc_tau = std::sin(tau) / tau;
        }
        rot(i, j) += cos_tau * W_matrix(i, k) * W_matrix(j, k);
        for (unsigned l = 0; l < n_orbs; l++) {
          rot(i, j) +=
              sinc_tau * W_matrix(i, k) * W_matrix(l, k) * X_matrix(l, j);
        }
      }
    }
  }
}

VectorXd FullOptimization::gradient(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) const {
  unsigned n_param = param_indices.size();
  VectorXd grad(n_param);
#pragma omp parallel for
  for (unsigned i = 0; i < n_param; i++) {
    unsigned p = param_indices[i].first;
    unsigned q = param_indices[i].second;
    grad(i) = 2 * (generalized_Fock(p, q) - generalized_Fock(q, p));
  }
  Timer::checkpoint("compute gradient");
  return grad;
}

MatrixXd FullOptimization::hessian(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) const {
  unsigned n_param = param_indices.size();
  MatrixXd hessian(n_param, n_param);
#pragma omp parallel for
  for (unsigned i = 0; i < n_param; i++) {
    for (unsigned j = 0; j <= i; j++) {
      unsigned p = param_indices[i].first;
      unsigned q = param_indices[i].second;
      unsigned r = param_indices[j].first;
      unsigned s = param_indices[j].second;
      hessian(i, j) = hessian_part(p, q, r, s) - hessian_part(p, q, s, r) -
                      hessian_part(q, p, r, s) + hessian_part(q, p, s, r);
      if (i != j)
        hessian(j, i) = hessian(i, j);
    }
  }
  Timer::checkpoint("compute hessian");
  return hessian;
}

VectorXd FullOptimization::hessian_diagonal(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) const {
  unsigned n_param = param_indices.size();
  VectorXd hessian_diagonal(n_param);
#pragma omp parallel for
  for (unsigned i = 0; i < n_param; i++) {
    unsigned p = param_indices[i].first;
    unsigned q = param_indices[i].second;
    hessian_diagonal(i) = hessian_part(p, q, p, q) - hessian_part(p, q, q, p) -
                          hessian_part(q, p, p, q) + hessian_part(q, p, q, p);
  }
  Timer::checkpoint("compute diagonal of hessian");
  return hessian_diagonal;
}

double FullOptimization::generalized_Fock(unsigned m, unsigned n) const {
  // Helgaker (10.8.24)
  double elem = 0.;
  for (unsigned q = 0; q < n_orbs; q++) {
    elem += one_rdm_elem(m, q) * integrals_p->get_1b(n, q);
  }
  for (unsigned q = 0; q < n_orbs; q++) {
    for (unsigned r = 0; r < n_orbs; r++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        elem +=
            two_rdm_elem(m, r, s, q) * integrals_p->get_2b(n, q, r, s);
      }
    }
  }
  return elem;
}

double FullOptimization::Y_matrix(unsigned p, unsigned q, unsigned r,
                              unsigned s) const {
  // Helgaker (10.8.50)
  double elem = 0.;
  for (unsigned m = 0; m < n_orbs; m++) {
    for (unsigned n = 0; n < n_orbs; n++) {
      elem +=
          (two_rdm_elem(p, r, n, m) + two_rdm_elem(p, n, r, m)) *
          integrals_p->get_2b(q, m, n, s);
      elem += two_rdm_elem(p, m, n, r) * integrals_p->get_2b(q, s, m, n);
    }
  }
  return elem;
}

double FullOptimization::hessian_part(unsigned p, unsigned q, unsigned r,
                                  unsigned s) const {
  // Helgaker (10.8.53) content in [...]
  double elem = 0.;
  elem += 2 * one_rdm_elem(p, r) * integrals_p->get_1b(q, s);
  if (q == s)
    elem -= (generalized_Fock(p, r) + generalized_Fock(r, p));
  elem += 2 * Y_matrix(p, q, r, s);
  return elem;
}
