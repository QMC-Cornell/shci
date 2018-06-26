#include "rdm.h"

#include <fgpl/src/hash_map.h>
#include <stdio.h>
#include <eigen/Eigen/Dense>
#include "../parallel.h"
#include "../timer.h"

void RDM::get_1rdm(
    const std::vector<Det>& dets, const std::vector<double>& coefs, const Integrals& integrals) {
  //=====================================================
  // Create 1RDM using the variational wavefunction
  //
  // Created: Y. Yao, June 2018
  //=====================================================

  unsigned n_orbs = integrals.n_orbs;
  unsigned n_up = integrals.n_up;
  unsigned n_dn = integrals.n_dn;
  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
  }

#pragma omp parallel for
  for (size_t idet = 0; idet < dets.size(); idet++) {
    Det this_det = dets[idet];

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
        if (det2coef.count(new_det) == 1)
          coef = det2coef[new_det];
        else
          continue;

#pragma omp atomic
        one_rdm(p, r) += this_det.up.diff(new_det.up).permutation_factor * coef * coefs[idet];
      }  // r
    }  // i_elec

    // dn electrons
    for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
      unsigned p = occ_dn[i_elec];
      for (unsigned r = 0; r < n_orbs; r++) {
        if (orb_sym[p] != orb_sym[r]) continue;
        if (p != r && this_det.dn.has(r)) continue;

        Det new_det = this_det;
        new_det.dn.unset(p);
        new_det.dn.set(r);

        double coef;
        if (det2coef.count(new_det) == 1)
          coef = det2coef[new_det];
        else
          continue;

#pragma omp atomic
        one_rdm(p, r) += this_det.dn.diff(new_det.dn).permutation_factor * coef * coefs[idet];
      }  // r
    }  // i_elec
  }  // idet
}

void RDM::generate_natorb_integrals(const Integrals& integrals) const {
  //======================================================
  // Compute natural orbitals by diagonalizing the 1RDM.
  // Rotate integrals to natural orbital basis and generate
  // new FCIDUMP file.
  // The original 1RDM (rdm) is left untouched.
  // This version stores the integrals in a 4D array
  //
  // Created: Y. Yao, June 2018
  //======================================================

  unsigned n_orbs = integrals.n_orbs;
  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  // Determine number of point gourp elements used for current system
  unsigned n_group_elements = orb_sym[1];
  for (size_t i = 1; i < orb_sym.size(); i++) {
    if (orb_sym[i] > n_group_elements) n_group_elements = orb_sym[i];
  }

  // Diagonalize rdm in the subspace of each irrep separately
  std::vector<std::vector<unsigned>> inds(n_group_elements);
  std::vector<unsigned> n_in_group(n_group_elements);

  for (unsigned irrep = 0; irrep < n_group_elements; irrep++) {
    n_in_group[irrep] = 0;
    for (unsigned i = 0; i < n_orbs; i++) {
      if (orb_sym[i] == irrep + 1) {
        n_in_group[irrep] = n_in_group[irrep] + 1;
        inds[irrep].push_back(i);
      }
    }
  }

  double eigenvalues[n_orbs];
  MatrixXd rot = MatrixXd::Zero(n_orbs, n_orbs);  // rotation matrix

  for (unsigned irrep = 0; irrep < n_group_elements; irrep++) {
    if (n_in_group[irrep] == 0) continue;
    unsigned n = n_in_group[irrep];

    MatrixXd tmp_rdm(n, n);  // rdm in the subspace of current irrep
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < n; j++) {
        tmp_rdm(i, j) = one_rdm(inds[irrep][i], inds[irrep][j]);
      }
    }

    SelfAdjointEigenSolver<MatrixXd> es(tmp_rdm);
    MatrixXd tmp_eigenvalues, tmp_eigenvectors;
    tmp_eigenvalues = es.eigenvalues().transpose();
    tmp_eigenvectors = es.eigenvectors();

    // columns of rot (rotation matrix) = eigenvectors of rdm
    for (unsigned i = 0; i < n; i++) {
      eigenvalues[inds[irrep][i]] = tmp_eigenvalues(n - i - 1);
      for (unsigned j = 0; j < n; j++) {
        rot(inds[irrep][i], inds[irrep][j]) = tmp_eigenvectors(i, n - j - 1);
      }
    }
  }  // irrep

  std::cout << "Occupation numbers:\n";
  for (unsigned i = 0; i < integrals.n_elecs && i < n_orbs; i++) {
    std::cout << eigenvalues[i] << "\n";
  }

  Timer::checkpoint("compute natural orbitals");

  // Rotate orbitals and generate new integrals
  std::vector<std::vector<std::vector<std::vector<double>>>> new_integrals(n_orbs);
  std::vector<std::vector<std::vector<std::vector<double>>>> tmp_integrals(n_orbs);

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
        std::fill(new_integrals[i][j][k].begin(), new_integrals[i][j][k].end(), 0.);
        std::fill(tmp_integrals[i][j][k].begin(), tmp_integrals[i][j][k].end(), 0.);
      }
    }
  }

// Two-body integrals
#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          tmp_integrals[p][q][r][s] = integrals.get_2b(p, q, r, s);
        }  // s
      }  // r
    }  // q
  }  // p

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
        }  // s
      }  // r
    }  // q
  }  // p

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
        }  // s
      }  // r
    }  // q
  }  // p

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
        }  // s
      }  // r
    }  // q
  }  // p

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
        }  // s
      }  // r
    }  // q
  }  // p

// One-body integrals
#pragma omp parallel for
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      tmp_integrals[p][q][n_orbs][n_orbs] = integrals.get_1b(p, q);
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

  Timer::checkpoint("computing new integrals");

  FILE* pFile;
  pFile = fopen("FCIDUMP_natorb", "w");

  // Header
  fprintf(pFile, "&FCI NORB=%d, NELEC=%d, MS2=%d,\n", n_orbs, integrals.n_elecs, 0);
  fprintf(pFile, "ORBSYM=");
  for (unsigned i = 0; i < n_orbs; i++) {
    fprintf(pFile, "  %d", orb_sym[integrals.orb_order_inv[i]]);
  }
  fprintf(pFile, "\nISYM=1\n&END\n");

  // Two-body integrals
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q <= p; q++) {
      for (unsigned r = 0; r <= p; r++) {
        for (unsigned s = 0; s <= r; s++) {
          if ((p == r) && (q < s)) continue;
          if (std::abs(new_integrals[p][q][r][s]) > 1e-8) {
            fprintf(
                pFile,
                " %19.12E %3d %3d %3d %3d\n",
                new_integrals[p][q][r][s],
                integrals.orb_order[p] + 1,
                integrals.orb_order[q] + 1,
                integrals.orb_order[r] + 1,
                integrals.orb_order[s] + 1);
          }
        }  // s
      }  // r
    }  // q
  }  // p

  // One-body integrals
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q <= p; q++) {
      if (std::abs(new_integrals[p][q][n_orbs][n_orbs]) > 1e-8) {
        fprintf(
            pFile,
            " %19.12E %3d %3d %3d %3d\n",
            new_integrals[p][q][n_orbs][n_orbs],
            integrals.orb_order[p] + 1,
            integrals.orb_order[q] + 1,
            0,
            0);
      }
    }
  }

  // Nuclear-nuclear energy
  fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", integrals.energy_core, 0, 0, 0, 0);

  fclose(pFile);

  Timer::checkpoint("creating new FCIDUMP");
}

/*
void RDM::generate_natorb_integrals(const Integrals& integrals) const {
//======================================================
// Compute natural orbitals by diagonalizing the 1RDM.
// Rotate integrals to natural orbital basis and generate
// new FCIDUMP file.
// The original 1RDM (rdm) is left untouched.
// This version stores the integrals in a hashtable
//
// Created: Y. Yao, June 2018
//======================================================
  unsigned n_orbs = integrals.n_orbs;
  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  // Determine number of point gourp elements used for current system
  unsigned n_group_elements = orb_sym[1];
  for (size_t i = 1; i < orb_sym.size(); i++) {
    if (orb_sym[i] > n_group_elements) n_group_elements = orb_sym[i];
  }

  // Diagonalize rdm in the subspace of each irrep separately
  std::vector<std::vector<unsigned>> inds(n_group_elements);
  std::vector<unsigned> n_in_group(n_group_elements);

  for (unsigned irrep = 0; irrep < n_group_elements; irrep++) {
    n_in_group[irrep] = 0;
    for (unsigned i=0; i<n_orbs; i++) {
      if (orb_sym[i]==irrep+1) {
        n_in_group[irrep] = n_in_group[irrep] + 1;
        inds[irrep].push_back(i);
      }
    }
  }

  double eigenvalues[n_orbs];
  MatrixXd rot = MatrixXd::Zero(n_orbs,n_orbs); // rotation matrix

  for (unsigned irrep=0; irrep<n_group_elements; irrep++) {
    if (n_in_group[irrep] == 0) continue;
    unsigned n = n_in_group[irrep];

    MatrixXd tmp_rdm(n,n); // rdm in the subspace of current irrep
    for (unsigned i=0; i<n; i++) {
      for (unsigned j=0; j<n; j++) {
        tmp_rdm(i,j) = one_rdm(inds[irrep][i],inds[irrep][j]);
      }
    }

    SelfAdjointEigenSolver<MatrixXd> es(tmp_rdm);
    MatrixXd tmp_eigenvalues, tmp_eigenvectors;
    tmp_eigenvalues = es.eigenvalues().transpose();
    tmp_eigenvectors = es.eigenvectors();

    // columns of rot (rotation matrix) = eigenvectors of rdm
    for (unsigned i=0; i<n; i++) {
      eigenvalues[inds[irrep][i]] = tmp_eigenvalues(n-i-1);
      for (unsigned j=0; j<n; j++) {
        rot(inds[irrep][i],inds[irrep][j]) = tmp_eigenvectors(i,n-j-1);
      }
    }
  } // irrep

  std::cout<< "Occupation numbers:\n";
  for (unsigned i=0; i<n_orbs;i++) std::cout<< eigenvalues[i] <<"\n";

  std::cout<<"Done computing natural orbitals. Computing new integrals."<<"\n";

  // Rotate orbitals and generate new integrals

  // Two-body integrals
  fgpl::HashMap<size_t, double, IntegralsHasher> new_integrals_2b, tmp_integrals_2b;

#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned r=0; r<n_orbs; r++) {
        for (unsigned s=0; s<n_orbs; s++) {
          double val_pqrs = integrals.get_2b(p,q,r,s);
          if (val_pqrs != 0.) tmp_integrals_2b.set(nonsym_combine4(p,q,r,s), val_pqrs);
//if (Parallel::is_master()) std::cout<<p<<" "<<q<<" "<<r<<" "<<s<<"\n";
        } // s
      } // r
    } // q
//std::cout<<p<<"\n";
  } // p
std::cout<<"BBB\n";
//#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned r=0; r<n_orbs; r++) {
        for (unsigned s=0; s<n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i=0; i<n_orbs; i++) {
            new_val += rot(i,p)*tmp_integrals_2b.get(nonsym_combine4(i,q,r,s),0.);
          }
          if (new_val != 0.) new_integrals_2b.set(nonsym_combine4(p,q,r,s), new_val);
        } // s
      } // r
    } // q
  } // p

  tmp_integrals_2b.clear();
  tmp_integrals_2b = new_integrals_2b;
std::cout<<"CCC\n";
//#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned r=0; r<n_orbs; r++) {
        for (unsigned s=0; s<n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i=0; i<n_orbs; i++) {
            new_val += rot(i,q)*tmp_integrals_2b.get(nonsym_combine4(p,i,r,s),0.);
          }
          if (new_val != 0.) new_integrals_2b.set(nonsym_combine4(p,q,r,s), new_val);
        } // s
      } // r
    } // q
  } // p

  tmp_integrals_2b.clear();
  tmp_integrals_2b = new_integrals_2b;
std::cout<<"DDD\n";
#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned r=0; r<n_orbs; r++) {
        for (unsigned s=0; s<n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i=0; i<n_orbs; i++) {
            new_val += rot(i,r)*tmp_integrals_2b.get(nonsym_combine4(p,q,i,s),0.);
          }
          if (new_val != 0.) new_integrals_2b.set(nonsym_combine4(p,q,r,s), new_val);
        } // s
      } // r
    } // q
  } // p

  tmp_integrals_2b.clear();
  tmp_integrals_2b = new_integrals_2b;
std::cout<<"EEE\n";
#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned r=0; r<n_orbs; r++) {
        for (unsigned s=0; s<n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i=0; i<n_orbs; i++) {
            new_val += rot(i,s)*tmp_integrals_2b.get(nonsym_combine4(p,q,r,i),0.);
          }
          if (new_val != 0.) new_integrals_2b.set(nonsym_combine4(p,q,r,s), new_val);
        } // s
      } // r
    } // q
  } // p

  tmp_integrals_2b.clear();

  // One-body integrals
  fgpl::HashMap<size_t, double, IntegralsHasher> new_integrals_1b, tmp_integrals_1b;

#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      double val_pq = integrals.get_1b(p,q);
      if (val_pq != 0.) tmp_integrals_1b.set(nonsym_combine2(p,q), val_pq);
    }
  }

#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i=0; i<n_orbs; i++) {
        new_val += rot(i,p)*tmp_integrals_1b.get(nonsym_combine2(i,q),0.);
      }
      if (new_val != 0.) new_integrals_1b.set(nonsym_combine2(p,q), new_val);
    }
  }


  tmp_integrals_1b.clear();
  tmp_integrals_1b = new_integrals_1b;

#pragma omp parallel for
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i=0; i<n_orbs; i++) {
        new_val += rot(i,q)*tmp_integrals_1b.get(nonsym_combine2(p,i),0.);
      }
      if (new_val != 0.) new_integrals_1b.set(nonsym_combine2(p,q), new_val);
    }
  }


  std::cout<< "Done computing new integrals. Dumping new integrals into file \n";


  FILE * pFile;
  pFile = fopen ("FCIDUMP_natorb","w");

  // Header
  fprintf(pFile, "&FCI NORB=%d, NELEC=%d, MS2=%d,\n", n_orbs, integrals.n_elecs, 0);
  fprintf(pFile, "ORBSYM=");
  for (unsigned i=0; i<n_orbs; i++) {
    fprintf(pFile, "  %d", orb_sym[integrals.orb_order_inv[i]]);
  }
  fprintf(pFile, "\nISYM=1\n&END\n");

  // Two-body integrals
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<=p; q++) {
      for (unsigned r=0; r<=p; r++) {
        for (unsigned s=0; s<=r; s++) {
          if ((p==r) && (q<s)) continue;
          double val_pqrs = new_integrals_2b.get(nonsym_combine4(p,q,r,s),0.);
          if (std::abs(val_pqrs) > 1e-8) {
            fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", val_pqrs, integrals.orb_order[p]+1,
integrals.orb_order[q]+1, integrals.orb_order[r]+1, integrals.orb_order[s]+1);
          }
        } // s
      } // r
    } // q
  } // p

  // One-body integrals
  for (unsigned p=0; p<n_orbs; p++) {
    for (unsigned q=0; q<=p; q++) {
      double val_pq = new_integrals_1b.get(nonsym_combine2(p,q),0.);
      if (std::abs(val_pq) > 1e-8) {
        fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", val_pq, integrals.orb_order[p]+1,
integrals.orb_order[q]+1, 0, 0);
      }
    }
  }

  // Nuclear-nuclear energy
  fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", integrals.energy_core, 0,0,0,0);

  fclose (pFile);

  std::cout<< "Done creating new FCIDUMP\n";
}


size_t RDM::nonsym_combine2(const size_t a, const size_t b) const {
  return a*n_orbs + b;
}

size_t RDM::nonsym_combine4(const size_t a, const size_t b, const size_t c, const size_t d) const {
  return nonsym_combine2(nonsym_combine2(nonsym_combine2(a,b),c),d);
}
*/

void RDM::get_2rdm(
    const std::vector<Det>& dets, const std::vector<double>& coefs, const Integrals& integrals) {
  //=====================================================
  // Create spatial 2RDM using the variational wavefunction
  //
  // D_pqrs corresponds to <a^+_p a^+_q a_r a_s>
  // Four different spin configurations
  // (1) p: up, q: up, r: up, s: up
  // (2) p: dn, q: dn, r: dn, s: dn
  // (3) p: dn, q: up, r: up, s: dn
  // (4) p: up, q: dn, r: dn, s: up
  //
  // Created: Y. Yao, June 2018
  //=====================================================

  unsigned n_orbs = integrals.n_orbs;
  unsigned n_up = integrals.n_up;
  unsigned n_dn = integrals.n_dn;
  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  std::vector<double> two_rdm(n_orbs * n_orbs * (n_orbs * n_orbs + 1) / 2);

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
  }

#pragma omp parallel for
  for (size_t idet = 0; idet < dets.size(); idet++) {
    Det this_det = dets[idet];
    double this_coef = coefs[idet];

    std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
    std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

    // (1) p: up, q: up, r: up, s: up
    for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
      Det new_det = this_det;
      unsigned s = occ_up[i_elec];
      new_det.up.unset(s);
      for (unsigned j_elec = i_elec + 1; j_elec < n_up; j_elec++) {
        unsigned r = occ_up[j_elec];
        new_det.up.unset(r);
        for (unsigned q = 0; q < n_orbs; q++) {
          if (new_det.up.has(q)) continue;
          new_det.up.set(q);
          for (unsigned p = 0; p < q; p++) {
            if (new_det.up.has(p)) continue;
            new_det.up.set(p);

            if (det2coef.count(new_det) == 1) {
              double coef = det2coef[new_det];
              double element = this_coef * coef * this_det.up.diff(new_det.up).permutation_factor;

#pragma omp atomic
              two_rdm[combine4_2rdm(p, q, r, s, n_orbs)] += element;
#pragma omp atomic
              two_rdm[combine4_2rdm(p, q, s, r, n_orbs)] -= element;  // since p<q, r>s
              // exclude (q,p,r,s,-=) and (q,p,s,r,+=) since we are taking advantage of
              // the symmetry (p,s) <-> (q,r) and only constructing half of the 2RDM
            }

            new_det.up.unset(p);
          }  // p
          new_det.up.unset(q);
        }  // r
        new_det.up.set(r);
      }  // j_elec
    }  // i_elec

    // (2) p: dn, q: dn, r: dn, s: dn
    for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
      Det new_det = this_det;
      unsigned s = occ_dn[i_elec];
      new_det.dn.unset(s);
      for (unsigned j_elec = i_elec + 1; j_elec < n_dn; j_elec++) {
        unsigned r = occ_dn[j_elec];
        new_det.dn.unset(r);
        for (unsigned q = 0; q < n_orbs; q++) {
          if (new_det.dn.has(q)) continue;
          new_det.dn.set(q);
          for (unsigned p = 0; p < q; p++) {
            if (new_det.dn.has(p)) continue;
            new_det.dn.set(p);

            if (det2coef.count(new_det) == 1) {
              double coef = det2coef[new_det];
              double element = this_coef * coef * this_det.dn.diff(new_det.dn).permutation_factor;

#pragma omp atomic
              two_rdm[combine4_2rdm(p, q, r, s, n_orbs)] += element;
#pragma omp atomic
              two_rdm[combine4_2rdm(p, q, s, r, n_orbs)] -= element;
            }

            new_det.dn.unset(p);
          }  // p
          new_det.dn.unset(q);
        }  // r
        new_det.dn.set(r);
      }  // j_elec
    }  // i_elec

    // (3) p: dn, q: up, r: up, s: dn
    // (4) p: up, q: dn, r: dn, s: up can be obtained from (3) by swapping indices
    for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
      Det new_det = this_det;
      unsigned s = occ_dn[i_elec];
      new_det.dn.unset(s);
      for (unsigned j_elec = 0; j_elec < n_up; j_elec++) {
        unsigned r = occ_up[j_elec];
        new_det.up.unset(r);
        for (unsigned q = 0; q < n_orbs; q++) {
          if (new_det.up.has(q)) continue;
          new_det.up.set(q);
          for (unsigned p = 0; p < n_orbs; p++) {
            if (new_det.dn.has(p)) continue;
            new_det.dn.set(p);

            if (det2coef.count(new_det) == 1) {
              double coef = det2coef[new_det];
              int perm_fac = this_det.up.diff(new_det.up).permutation_factor *
                             this_det.dn.diff(new_det.dn).permutation_factor;
              double element = this_coef * coef * perm_fac;

#pragma omp atomic
              two_rdm[combine4_2rdm(p, q, r, s, n_orbs)] += element;
              if (p == q && s == r) {
#pragma omp atomic              
                two_rdm[combine4_2rdm(q, p, s, r, n_orbs)] += element;                
              }
              // this line needed as a result of storing only half of 2RDM, (p,s) = (q,r)
            }

            new_det.dn.unset(p);
          }  // p
          new_det.up.unset(q);
        }  // r
        new_det.up.set(r);
      }  // j_elec
    }  // i_elec

  }  // idet
  /*
  std::cout<<"before print\n";
  for (unsigned p =0; p<n_orbs; p++) {
    for (unsigned q=0; q<n_orbs; q++) {
      for (unsigned s=0; s<n_orbs; s++) {
        for (unsigned r=0; r<n_orbs; r++) {
  if (std::abs(two_rdm[combine4_2rdm(p,q,r,s,n_orbs)]) > 1.e-6)
  std::cout<<p<<"\t"<<q<<"\t"<<s<<"\t"<<r<<"\t"<<two_rdm[combine4_2rdm(p,q,r,s,n_orbs)]<<"\n";
        } //r
      } //s
    }  //q
  } // p
  */
  std::cout << "writing out 2RDM\n";

  FILE* pFile;
  pFile = fopen("spatialRDM.txt", "w");

  fprintf(pFile, "%d\n", n_orbs);
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        for (unsigned r = 0; r < n_orbs; r++) {
          if (std::abs(two_rdm[combine4_2rdm(p, q, r, s, n_orbs)]) > 1.e-6)
            fprintf(
                pFile,
                "%3d   %3d   %3d   %3d   %10.8g\n",
                p,
                q,
                s,
                r,
                two_rdm[combine4_2rdm(p, q, r, s, n_orbs)]);
        }  // r
      }  // s
    }  // q
  }  // p
}

unsigned RDM::combine4_2rdm(unsigned p, unsigned q, unsigned r, unsigned s, unsigned n_orbs) const {
  unsigned a = p * n_orbs + s;
  unsigned b = q * n_orbs + r;
  if (a > b) {
    return (a * (a + 1)) / 2 + b;
  } else {
    return (b * (b + 1)) / 2 + a;
  }
}
