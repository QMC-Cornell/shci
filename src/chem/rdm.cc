#include "rdm.h"

#include <fgpl/src/hash_map.h>
#include <math.h>
#include <stdio.h>
#include <eigen/Eigen/Dense>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"

void RDM::get_1rdm(
    const std::vector<Det>& dets,
    const std::vector<double>& coefs,
    const Integrals& integrals,
    const bool dump_csv) {
  //=====================================================
  // Create 1RDM using the variational wavefunction
  //
  // Created: Y. Yao, June 2018
  //=====================================================
  bool time_sym = Config::get<bool>("time_sym", false);

  n_orbs = integrals.n_orbs;
  n_up = integrals.n_up;
  n_dn = integrals.n_dn;

  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
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
        if (det2coef.count(new_det) == 1)
          coef = det2coef[new_det];
        else
          continue;

#pragma omp atomic
        one_rdm(p, r) += this_det.up.diff(new_det.up).permutation_factor * coef * coefs[i_det];
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
          if (det2coef.count(new_det) == 1)
            coef = det2coef[new_det];
          else
            continue;

#pragma omp atomic
          one_rdm(p, r) += this_det.dn.diff(new_det.dn).permutation_factor * coef * coefs[i_det];
        }  // r
      }  // i_elec
    }
  }  // i_det

  if (time_sym) one_rdm *= 2.;

  if (dump_csv) {
    FILE* pFile;
    pFile = fopen("1rdm.csv", "w");
    fprintf(pFile, "p,r,1rdm\n");
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned r = p; r < n_orbs; r++) {
        const double rdm_pr = one_rdm(p, r);
        if (std::abs(rdm_pr) < 1e-9) continue;
        fprintf(pFile, "%d,%d,%#.15g\n", integrals.orb_order[p], integrals.orb_order[r], rdm_pr);
      }
    }
    fclose(pFile);
  }
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

void RDM::get_2rdm_slow(
    const std::vector<Det>& dets, const std::vector<double>& coefs, const Integrals& integrals) {
  //=====================================================
  // Create spatial 2RDM using the variational wavefunction.
  // This version does not make use of Hamiltonian connections
  // and is slow for chemistry systems.
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
  // When time_sym is used, the input wavefunction should
  // be unpacked before passing in.
  //
  // Created: Y. Yao, June 2018
  //=====================================================

  unsigned n_orbs = integrals.n_orbs;
  unsigned n_up = integrals.n_up;
  unsigned n_dn = integrals.n_dn;
  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  two_rdm.resize((n_orbs * n_orbs * (n_orbs * n_orbs + 1) / 2), 0.);

  // Create hash table; used for looking up the coef of a det
  std::unordered_map<Det, double, DetHasher> det2coef;
  for (size_t i = 0; i < dets.size(); i++) {
    det2coef[dets[i]] = coefs[i];
  }

#pragma omp parallel for
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    Det this_det = dets[i_det];
    double this_coef = coefs[i_det];

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
              double element = this_coef * coef * permfac_ccaa(this_det.up, p, q, r, s);
              if (Config::get<bool>("time_sym", false)) element *= 2.;

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

    if (!Config::get<bool>("time_sym", false)) {
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
                double element = this_coef * coef * permfac_ccaa(this_det.dn, p, q, r, s);

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
    }

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

  }  // i_det

  Timer::checkpoint("computing 2RDM");

  std::cout << "writing out 2RDM\n";
 
  // csv format
  FILE* pFile = fopen("2rdm.csv", "w");
  fprintf(pFile, "p,q,r,s,2rdm\n");
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = p; q < n_orbs; q++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        for (unsigned r = 0; r < n_orbs; r++) {
          if (p == q && s > r) continue;
          const double rdm_pqrs = two_rdm[combine4_2rdm(p, q, r, s, n_orbs)];
          if (std::abs(rdm_pqrs) < 1.0e-9) continue;
          fprintf(
              pFile,
              "%d,%d,%d,%d,%#.15g\n",
              integrals.orb_order[p],
              integrals.orb_order[q],
              integrals.orb_order[r],
              integrals.orb_order[s],
              rdm_pqrs);
        }
      }
    }
  }
  fclose(pFile);

  // txt format
  /*
  FILE* pFile;
  pFile = fopen("spatialRDM.txt", "w");

  fprintf(pFile, "%d\n", n_orbs);

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        for (unsigned r = 0; r < n_orbs; r++) {
          if (std::abs(
                  two_rdm[combine4_2rdm(
                      integrals.orb_order_inv[p],
                      integrals.orb_order_inv[q],
                      integrals.orb_order_inv[r],
                      integrals.orb_order_inv[s],
                      n_orbs)]) > 1.e-6)
            fprintf(
                pFile,
                "%3d   %3d   %3d   %3d   %10.8g\n",
                p,
                q,
                s,
                r,
                two_rdm[combine4_2rdm(
                    integrals.orb_order_inv[p],
                    integrals.orb_order_inv[q],
                    integrals.orb_order_inv[r],
                    integrals.orb_order_inv[s],
                    n_orbs)]);
        }  // r
      }  // s
    }  // q
  }  // p
  */
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

int RDM::permfac_ccaa(HalfDet halfket, unsigned p, unsigned q, unsigned r, unsigned s) const {
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

void RDM::get_2rdm(
    const std::vector<Det>& dets,
    const std::vector<double>& coefs,
    const Integrals& integrals,
    const std::vector<std::vector<size_t>>& connections,
    const bool dump_csv) {
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
  //=====================================================
  bool time_sym = Config::get<bool>("time_sym", false);

  n_orbs = integrals.n_orbs;
  n_up = integrals.n_up;
  n_dn = integrals.n_dn;

  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  two_rdm.resize((n_orbs * n_orbs * (n_orbs * n_orbs + 1) / 2), 0.);

#pragma omp parallel for schedule(dynamic, 5)
  for (size_t i_det = 0; i_det < connections.size(); i_det++) {
    Det this_det = dets[i_det];
    double this_coef = coefs[i_det];

    for (size_t j_det = 0; j_det < connections[i_det].size(); j_det++) {
      size_t connected_ind = connections[i_det][j_det];
      Det connected_det = dets[connected_ind];
      double connected_coef = coefs[connected_ind];

      if (!time_sym)
        get_2rdm_elements(connected_det, connected_coef, this_det, this_coef);
      else {
        if (this_det.up == this_det.dn) {
          if (connected_det.up == connected_det.dn) {
            get_2rdm_elements(connected_det, connected_coef, this_det, this_coef);
          } else {
            Det connected_det_rev = connected_det;
            connected_det_rev.reverse_spin();
            double connected_coef_new = connected_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef_new, this_det, this_coef);
            get_2rdm_elements(connected_det_rev, connected_coef_new, this_det, this_coef);
          }
        } else {
          if (connected_det.up == connected_det.dn) {
            Det this_det_rev = this_det;
            this_det_rev.reverse_spin();
            double this_coef_new = this_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef, this_det, this_coef_new);
            get_2rdm_elements(connected_det, connected_coef, this_det_rev, this_coef_new);
          } else {
            Det connected_det_rev = connected_det;
            connected_det_rev.reverse_spin();
            double connected_coef_new = connected_coef * Util::SQRT2_INV;
            Det this_det_rev = this_det;
            this_det_rev.reverse_spin();
            double this_coef_new = this_coef * Util::SQRT2_INV;

            get_2rdm_elements(connected_det, connected_coef_new, this_det, this_coef_new);
            if (j_det != 0)
              get_2rdm_elements(connected_det, connected_coef_new, this_det_rev, this_coef_new);
            get_2rdm_elements(connected_det_rev, connected_coef_new, this_det, this_coef_new);
            get_2rdm_elements(connected_det_rev, connected_coef_new, this_det_rev, this_coef_new);
          }
        }
      }
    }
  }

  Timer::checkpoint("computing 2RDM");

  std::cout << "writing out 2RDM\n";

  if (dump_csv) {
    FILE* pFile = fopen("2rdm.csv", "w");
    fprintf(pFile, "p,q,r,s,2rdm\n");
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned q = p; q < n_orbs; q++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          for (unsigned r = 0; r < n_orbs; r++) {
            if (p == q && s > r) continue;
            const double rdm_pqrs = two_rdm[combine4_2rdm(p, q, r, s, n_orbs)];
            if (std::abs(rdm_pqrs) < 1.0e-9) continue;
            fprintf(
                pFile,
                "%d,%d,%d,%d,%#.15g\n",
                integrals.orb_order[p],
                integrals.orb_order[q],
                integrals.orb_order[r],
                integrals.orb_order[s],
                rdm_pqrs);
          }
        }
      }
    }
    fclose(pFile);
  } else {
    FILE* pFile;
    pFile = fopen("spatialRDM.txt", "w");

    fprintf(pFile, "%d\n", n_orbs);

    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned q = 0; q < n_orbs; q++) {
        for (unsigned s = 0; s < n_orbs;
             s++) {  // r and s switched in keeping with Dice conventions
          for (unsigned r = 0; r < n_orbs; r++) {
            if (std::abs(
                    two_rdm[combine4_2rdm(
                        integrals.orb_order_inv[p],
                        integrals.orb_order_inv[q],
                        integrals.orb_order_inv[r],
                        integrals.orb_order_inv[s],
                        n_orbs)]) > 1.e-6)
              fprintf(
                  pFile,
                  "%3d   %3d   %3d   %3d   %10.8g\n",
                  p,
                  q,
                  s,
                  r,
                  two_rdm[combine4_2rdm(
                      integrals.orb_order_inv[p],
                      integrals.orb_order_inv[q],
                      integrals.orb_order_inv[r],
                      integrals.orb_order_inv[s],
                      n_orbs)]);
          }  // r
        }  // s
      }  // q
    }  // p
  }
}

void RDM::get_2rdm_elements(
    const Det& connected_det,
    const double& connected_coef,
    const Det& this_det,
    const double& this_coef) {
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
          double element = this_coef * connected_coef;

          write_in_2rdm(s, r, r, s, element);
          write_in_2rdm(s, r, s, r, -element);
          write_in_2rdm(r, s, s, r, element);
          write_in_2rdm(r, s, r, s, -element);
        }
      }

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        for (unsigned j_elec = i_elec + 1; j_elec < n_dn; j_elec++) {
          unsigned s = occ_dn[i_elec];
          unsigned r = occ_dn[j_elec];
          double element = this_coef * connected_coef;

          write_in_2rdm(s, r, r, s, element);
          write_in_2rdm(s, r, s, r, -element);
          write_in_2rdm(r, s, s, r, element);
          write_in_2rdm(r, s, r, s, -element);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        for (unsigned j_elec = 0; j_elec < n_dn; j_elec++) {
          unsigned s = occ_up[i_elec];
          unsigned r = occ_dn[j_elec];
          double element = this_coef * connected_coef;

          write_in_2rdm(s, r, r, s, element);
          write_in_2rdm(r, s, s, r, element);
        }
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // to

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        if ((p != b1) && (p != b2)) {
          double element = this_coef * connected_coef * permfac_ccaa(this_det.dn, p, b2, b1, p);

          write_in_2rdm(p, b2, b1, p, element);
          write_in_2rdm(b2, p, b1, p, -element);
          write_in_2rdm(p, b2, p, b1, -element);
          write_in_2rdm(b2, p, p, b1, element);

          element = this_coef * connected_coef * permfac_ccaa(connected_det.dn, p, b1, b2, p);

          write_in_2rdm(p, b1, b2, p, element);
          write_in_2rdm(b1, p, b2, p, -element);
          write_in_2rdm(p, b1, p, b2, -element);
          write_in_2rdm(b1, p, p, b2, element);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up[i_elec];
        double element =
            this_coef * connected_coef * connected_det.dn.diff(this_det.dn).permutation_factor;

        write_in_2rdm(p, b2, b1, p, element);
        write_in_2rdm(b2, p, p, b1, element);

        element =
            this_coef * connected_coef * this_det.dn.diff(connected_det.dn).permutation_factor;

        write_in_2rdm(p, b1, b2, p, element);
        write_in_2rdm(b1, p, p, b2, element);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 2) {  // 2 beta excitations apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).right_only[1];
      unsigned b3 = connected_det.dn.diff(this_det.dn).left_only[0];  // to
      unsigned b4 = connected_det.dn.diff(this_det.dn).left_only[1];

      double element =
          this_coef * connected_coef * connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm(b3, b4, b2, b1, element);
      write_in_2rdm(b3, b4, b1, b2, -element);
      write_in_2rdm(b4, b3, b2, b1, -element);
      write_in_2rdm(b4, b3, b1, b2, element);

      element = this_coef * connected_coef * this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm(b1, b2, b4, b3, element);
      write_in_2rdm(b1, b2, b3, b4, -element);
      write_in_2rdm(b2, b1, b4, b3, -element);
      write_in_2rdm(b2, b1, b3, b4, element);
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
          double element = this_coef * connected_coef * permfac_ccaa(this_det.up, p, a2, a1, p);

          write_in_2rdm(p, a2, a1, p, element);
          write_in_2rdm(p, a2, p, a1, -element);
          write_in_2rdm(a2, p, a1, p, -element);
          write_in_2rdm(a2, p, p, a1, element);

          element = this_coef * connected_coef * permfac_ccaa(connected_det.up, p, a1, a2, p);

          write_in_2rdm(p, a1, a2, p, element);
          write_in_2rdm(p, a1, p, a2, -element);
          write_in_2rdm(a1, p, a2, p, -element);
          write_in_2rdm(a1, p, p, a2, element);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];

        double element =
            this_coef * connected_coef * connected_det.up.diff(this_det.up).permutation_factor;

        write_in_2rdm(a2, p, p, a1, element);
        write_in_2rdm(p, a2, a1, p, element);

        element =
            this_coef * connected_coef * this_det.up.diff(connected_det.up).permutation_factor;

        write_in_2rdm(a1, p, p, a2, element);
        write_in_2rdm(p, a1, a2, p, element);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // up from
      unsigned a2 = connected_det.up.diff(this_det.up).left_only[0];  // up to
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // dn from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // dn to

      double element = this_coef * connected_coef *
                       connected_det.up.diff(this_det.up).permutation_factor *
                       connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm(a2, b2, b1, a1, element);
      write_in_2rdm(b2, a2, a1, b1, element);

      element = this_coef * connected_coef * this_det.up.diff(connected_det.up).permutation_factor *
                this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm(a1, b1, b2, a2, element);
      write_in_2rdm(b1, a1, a2, b2, element);
    }

    // 2 alpha excitations apart
  } else if (connected_det.up.diff(this_det.up).n_diffs == 2) {
    if (this_det.dn == connected_det.dn) {
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // from
      unsigned a2 = connected_det.up.diff(this_det.up).right_only[1];
      unsigned a3 = connected_det.up.diff(this_det.up).left_only[0];  // to
      unsigned a4 = connected_det.up.diff(this_det.up).left_only[1];

      double element =
          this_coef * connected_coef * connected_det.up.diff(this_det.up).permutation_factor;

      write_in_2rdm(a3, a4, a2, a1, element);
      write_in_2rdm(a3, a4, a1, a2, -element);
      write_in_2rdm(a4, a3, a2, a1, -element);
      write_in_2rdm(a4, a3, a1, a2, element);

      element = this_coef * connected_coef * this_det.up.diff(connected_det.up).permutation_factor;

      write_in_2rdm(a1, a2, a4, a3, element);
      write_in_2rdm(a1, a2, a3, a4, -element);
      write_in_2rdm(a2, a1, a4, a3, -element);
      write_in_2rdm(a2, a1, a3, a4, element);
    }
  }
}

void RDM::write_in_2rdm(unsigned p, unsigned q, unsigned r, unsigned s, double value) {
  // By symmetry (p,s)<->(q,r) only half of the 2RDM needs storing.
  unsigned a = p * n_orbs + s;
  unsigned b = q * n_orbs + r;
  if (a >= b)
#pragma omp atomic
    two_rdm[(a * (a + 1)) / 2 + b] += value;
}

void RDM::compute_energy_from_rdm(const Integrals& integrals) const {
  //=====================================================
  // Reproduce variational energy from 2RDM.
  // Currently not used in the program;
  // Useful tool for verifying the 2RDM.
  //
  // Created: Y. Yao, July 2018
  //=====================================================

  unsigned n_orbs = integrals.n_orbs;
  unsigned n_up = integrals.n_up;
  unsigned n_dn = integrals.n_dn;

  MatrixXd tmp_1rdm = MatrixXd::Zero(n_orbs, n_orbs);

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned s = 0; s < n_orbs; s++) {
      for (unsigned k = 0; k < n_orbs; k++) {
        tmp_1rdm(p, s) += two_rdm[combine4_2rdm(p, k, k, s, n_orbs)] / (1. * (n_up + n_dn) - 1.);
      }
    }
  }

  std::cout << "tmp_1rdm\n" << tmp_1rdm << "\n";
  std::cout << "one_rdm\n" << one_rdm << "\n";

  double onebody = 0., twobody = 0.;

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      onebody += tmp_1rdm(p, q) * integrals.get_1b(p, q);
    }
  }

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          twobody +=
              0.5 * two_rdm[combine4_2rdm(p, q, r, s, n_orbs)] * integrals.get_2b(p, s, q, r);
        }
      }
    }
  }

  std::cout << "core energy: " << integrals.energy_core << "\none-body energy: " << onebody
            << "\ntwo-body energy: " << twobody
            << "\ntotal-energy: " << onebody + twobody + integrals.energy_core << "\n";
}
