#include "optimization.h"

#include "../parallel.h"
#include "../timer.h"

void Optimization::generate_natorb_integrals() {
  //======================================================
  // Compute natural orbitals by diagonalizing the 1RDM.
  // Rotate integrals to natural orbital basis and generate
  // new FCIDUMP file.
  // This version stores the integrals in a 4D array
  //
  // Created: Y. Yao, June 2018
  //======================================================

  std::vector<unsigned int> orb_sym = integrals_p->orb_sym;

  // Determine number of point group elements used for current system
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
        n_in_group[irrep] += 1;
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
        tmp_rdm(i, j) = rdm_p->one_rdm_elem(inds[irrep][i], inds[irrep][j]);
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

  rdm_p->clear();

  if (Parallel::is_master()) {
    std::cout << "Occupation numbers:\n";
    for (unsigned i = 0; i < integrals_p->n_elecs && i < n_orbs; i++) {
      std::cout << eigenvalues[i] << "\n";
    }
  }
  Timer::checkpoint("compute natural orbitals");

  rotate_integrals(rot);
  // dump_integrals("FCIDUMP_natorb");
}

void Optimization::rotate_integrals(const MatrixXd& rot) {
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
          tmp_integrals[p][q][r][s] = integrals_p->get_2b(p, q, r, s);
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

void Optimization::dump_integrals(const char* file_name) const {
  if (Parallel::is_master()) {
    FILE* pFile;
    pFile = fopen(file_name, "w");

    // Header
    fprintf(pFile, "&FCI NORB=%d, NELEC=%d, MS2=%d,\n", n_orbs, integrals_p->n_elecs, 0);
    fprintf(pFile, "ORBSYM=");
    for (unsigned i = 0; i < n_orbs; i++) {
      fprintf(pFile, "  %d", integrals_p->orb_sym[i]);
    }
    fprintf(pFile, "\nISYM=1\n&END\n");

    double integral_value;

    // Two-body integrals
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned q = 0; q <= p; q++) {
        for (unsigned r = 0; r <= p; r++) {
          for (unsigned s = 0; s <= r; s++) {
            if ((p == r) && (q < s)) continue;
            integral_value = new_integrals[p][q][r][s];
            // integral_value = integrals_p->get_2b(p, q, r, s);
            if (std::abs(integral_value) > 1e-9) {
              fprintf(
                  pFile,
                  " %19.12E %3d %3d %3d %3d\n",
                  integral_value,
                  integrals_p->orb_order[p] + 1,
                  integrals_p->orb_order[q] + 1,
                  integrals_p->orb_order[r] + 1,
                  integrals_p->orb_order[s] + 1);
            }
          }  // s
        }  // r
      }  // q
    }  // p

    // One-body integrals
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned q = 0; q <= p; q++) {
        integral_value = new_integrals[p][q][n_orbs][n_orbs];
        // integral_value = integrals_p->get_1b(p, q);
        if (std::abs(integral_value) > 1e-9) {
          fprintf(
              pFile,
              " %19.12E %3d %3d %3d %3d\n",
              integral_value,
              integrals_p->orb_order[p] + 1,
              integrals_p->orb_order[q] + 1,
              0,
              0);
        }
      }
    }

    // Nuclear-nuclear energy
    fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", integrals_p->energy_core, 0, 0, 0, 0);

    fclose(pFile);
  }

  Timer::checkpoint("creating new FCIDUMP");
}

void Optimization::rewrite_integrals() {
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
          integrals_p->integrals_2b.set(
              Integrals::combine4(p, q, r, s), value, [&](double& a, const double& b) {
                if (std::abs(a) < std::abs(b)) a = b;
              });
        }
      }
    }
  }

  for (p = 0; p < n_orbs; p++) {
    for (q = 0; q < n_orbs; q++) {
      value = new_integrals[p][q][n_orbs][n_orbs];
      integrals_p->integrals_1b.set(
          Integrals::combine2(p, q), value, [&](double& a, const double& b) {
            if (std::abs(a) < std::abs(b)) a = b;
          });
    }
  }
}

void Optimization::generate_optorb_integrals_from_newton() {
  std::vector<index_t> param_indices = parameter_indices();
  VectorXd grad = gradient(param_indices);
  std::cout << "\ngrad norm " << grad.norm();

  MatrixXd hess = hessian(param_indices);
  rdm_p->clear();

  /*
  std::cout << "\n gradient: ";
  for (unsigned i = 0; i < param_indices.size(); i++) {
    std::cout << grad(i) << " ";
  }*/

  // rotation matrix
  VectorXd new_param = hess.householderQr().solve(-1 * grad);

  MatrixXd rot = MatrixXd::Zero(n_orbs, n_orbs);
  fill_rot_matrix_with_parameters(rot, new_param, param_indices);
  rotate_integrals(rot);
}

void Optimization::generate_optorb_integrals_from_approximate_newton() {
  std::vector<index_t> param_indices = parameter_indices();
  VectorXd grad = gradient(param_indices);

  //std::cout << "\ngrad \n" << grad << "\n";
  //std::cout << "\ngrad norm " << grad.norm() << "\n";

  MatrixXd hess_diag = hessian_diagonal(param_indices);
  
  /*
  std::cout << "\n gradient: ";
  for (unsigned i = 0; i < param_indices.size(); i++) {
    std::cout << grad(i) << " ";
  }
  std::cout << "\n hessian diagonal: ";
  for (unsigned i = 0; i < param_indices.size(); i++) {
    std::cout << hess_diag(i) << " ";
  }
  */

  rdm_p->clear();
  VectorXd new_param(param_indices.size());
  for (unsigned i = 0; i < param_indices.size(); i++) {
    if (hess_diag(i) > 1e-5) {
      new_param(i) = -grad(i) / hess_diag(i);
    } else {
      new_param(i) = 0.;
    }
  }

  MatrixXd rot = MatrixXd::Zero(n_orbs, n_orbs);
  fill_rot_matrix_with_parameters(rot, new_param, param_indices);
  rotate_integrals(rot);
}

void Optimization::generate_optorb_integrals_from_grad_descent() {
  std::vector<index_t> param_indices = parameter_indices();

  VectorXd grad = gradient(param_indices);
  // std::cout << "\ngrad \n" << grad;
  // std::cout << "\ngrad norm " << grad.norm();
  rdm_p->clear();

  VectorXd new_param = -0.01 * grad;  // old_parameters are all zeros by defn
  MatrixXd rot = MatrixXd::Zero(n_orbs, n_orbs);
  fill_rot_matrix_with_parameters(rot, new_param, param_indices);
  rotate_integrals(rot);
}

std::vector<std::vector<double>>& Optimization::generate_optorb_integrals_from_adadelta(
    std::vector<std::vector<double>>& history) {
  std::vector<index_t> param_indices = parameter_indices();
  unsigned dim = param_indices.size();

  if (history[0].size() == 0) {
    history[0].resize(dim, 0.);  // E[update^2]
    history[1].resize(dim, 0.);  // E[g^2]
  }

  VectorXd grad = gradient(param_indices);
  // std::cout << "\ngrad \n" << grad;
  rdm_p->clear();

  double eps = 1e-8;
  double gamma = Config::get<double>("optimization/parameters/gamma", 0.9);
  if (Parallel::is_master()) {
    std::cout << "learning parameter: gamma = " << gamma << "\n";
  }
  VectorXd new_param = VectorXd::Zero(dim);
  double learning_rate, update;
  for (unsigned i = 0; i < dim; i++) {
    learning_rate = std::sqrt(history[0][i] + eps) /
                    std::sqrt(gamma * history[1][i] + (1. - gamma) * std::pow(grad(i), 2) + eps);
    update = -learning_rate * grad(i);
    // std::cout << "\nlearning rate " << learning_rate;
    new_param(i) = update;
    history[0][i] = gamma * history[0][i] + (1. - gamma) * std::pow(update, 2);
    history[1][i] = gamma * history[1][i] + (1. - gamma) * std::pow(grad(i), 2);
  }
  MatrixXd rot = MatrixXd::Zero(dim, dim);
  fill_rot_matrix_with_parameters(rot, new_param, param_indices);
  rotate_integrals(rot);

  return history;
}

std::vector<std::vector<double>>& Optimization::generate_optorb_integrals_from_amsgrad(
    std::vector<std::vector<double>>& history) {
  std::vector<index_t> param_indices = parameter_indices();
  unsigned dim = param_indices.size();

  if (history[0].size() == 0) {
    history[0].resize(dim, 0.);  // m
    history[1].resize(dim, 0.);  // v_hat
  }

  VectorXd grad = gradient(param_indices);
  // std::cout << "\ngrad \n" << grad;
  rdm_p->clear();

  double eps = 1e-8;
  double eta = Config::get<double>("optimization/parameters/eta", 0.01);
  double beta1 = Config::get<double>("optimization/parameters/beta1", 0.5);
  double beta2 = Config::get<double>("optimization/parameters/beta2", 0.5);

  if (Parallel::is_master()) {
    std::cout << "learning parameters: eta, beta1, beta2 = " << eta << ", " << beta1 << ", "
              << beta2 << "\n";
  }

  VectorXd new_param = VectorXd::Zero(dim);
  double m_prev, v_hat_prev, m, v, v_hat;
  for (unsigned i = 0; i < dim; i++) {
    m_prev = history[0][i];
    v_hat_prev = history[1][i];
    m = beta1 * m_prev + (1 - beta1) * grad(i);
    v = beta2 * v_hat_prev + (1 - beta2) * std::pow(grad(i), 2);
    v_hat = (v_hat_prev > v ? v_hat_prev : v);
    new_param(i) = -eta / (std::sqrt(v_hat) + eps) * m;
    history[0][i] = m;
    history[1][i] = v_hat;
  }

  MatrixXd rot = MatrixXd::Zero(dim, dim);
  fill_rot_matrix_with_parameters(rot, new_param, param_indices);
  rotate_integrals(rot);

  return history;
}

std::vector<Optimization::index_t> Optimization::parameter_indices() const {
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
  return indices;
}

void Optimization::fill_rot_matrix_with_parameters(
    MatrixXd& rot, const VectorXd& parameters, const std::vector<index_t>& parameter_indices) {
  // fill in the rot matrix with parameters, with entry indices being parameter_indices
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
  Tau2 = es.eigenvalues().transpose();  // Tau^2
  W_matrix = es.eigenvectors();
  /*
  std::cout << "\neigenval:\n";
  for (unsigned i = 0; i < n_orbs; i++) std::cout << Tau2(i) << " ";
  std::cout << "\n";
  */

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
          rot(i, j) += sinc_tau * W_matrix(i, k) * W_matrix(l, k) * X_matrix(l, j);
        }
      }
    }
  }
}

VectorXd Optimization::gradient(
    const std::vector<std::pair<unsigned, unsigned>>& param_indices) const {
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

MatrixXd Optimization::hessian(
    const std::vector<std::pair<unsigned, unsigned>>& param_indices) const {
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
      if (i != j) hessian(j, i) = hessian(i, j);
    }
  }
  Timer::checkpoint("compute hessian");
  return hessian;
}

VectorXd Optimization::hessian_diagonal(
    const std::vector<std::pair<unsigned, unsigned>>& param_indices) const {
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

double Optimization::generalized_Fock(unsigned m, unsigned n) const {
  // Helgaker (10.8.24)
  double elem = 0.;
  for (unsigned q = 0; q < n_orbs; q++) {
    elem += rdm_p->one_rdm_elem(m, q) * integrals_p->get_1b(n, q);
  }
  for (unsigned q = 0; q < n_orbs; q++) {
    for (unsigned r = 0; r < n_orbs; r++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        elem += rdm_p->two_rdm_elem(m, r, s, q) * integrals_p->get_2b(n, q, r, s);
      }
    }
  }
  return elem;
}

double Optimization::Y_matrix(unsigned p, unsigned q, unsigned r, unsigned s) const {
  // Helgaker (10.8.50)
  double elem = 0.;
  for (unsigned m = 0; m < n_orbs; m++) {
    for (unsigned n = 0; n < n_orbs; n++) {
      elem += (rdm_p->two_rdm_elem(p, r, n, m) + rdm_p->two_rdm_elem(p, n, r, m)) *
              integrals_p->get_2b(q, m, n, s);
      elem += rdm_p->two_rdm_elem(p, m, n, r) * integrals_p->get_2b(q, s, m, n);
    }
  }
  return elem;
}

double Optimization::hessian_part(unsigned p, unsigned q, unsigned r, unsigned s) const {
  // Helgaker (10.8.53) content in [...]
  double elem = 0.;
  elem += 2 * rdm_p->one_rdm_elem(p, r) * integrals_p->get_1b(q, s);
  if (q == s) elem -= (generalized_Fock(p, r) + generalized_Fock(r, p));
  elem += 2 * Y_matrix(p, q, r, s);
  return elem;
}
