#include "optimization.h"

#include "../parallel.h"
#include "../timer.h"

void Optimization::get_natorb_rotation_matrix() {
  //======================================================
  // Compute natural orbitals by diagonalizing the 1RDM.
  // Rotate integrals to natural orbital basis and generate
  // new FCIDUMP file.
  // This version stores the integrals in a 4D array
  //
  // Created: Y. Yao, June 2018
  //======================================================

  rdm.get_1rdm();

  std::vector<unsigned int> orb_sym = integrals.orb_sym;

  // Determine number of point group elements used for current system
  unsigned n_group_elements = orb_sym[1];
  for (size_t i = 1; i < orb_sym.size(); i++) {
    if (orb_sym[i] > n_group_elements)
      n_group_elements = orb_sym[i];
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
  // MatrixXd rot = MatrixXd::Zero(n_orbs, n_orbs);  // rotation matrix

  for (unsigned irrep = 0; irrep < n_group_elements; irrep++) {
    if (n_in_group[irrep] == 0)
      continue;
    unsigned n = n_in_group[irrep];

    MatrixXd tmp_rdm(n, n); // rdm in the subspace of current irrep
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < n; j++) {
        tmp_rdm(i, j) = rdm.one_rdm_elem(inds[irrep][i], inds[irrep][j]);
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
  } // irrep

  rdm.clear();

  if (Parallel::is_master()) {
    std::cout << "Occupation numbers:\n";
    for (unsigned i = 0; i < integrals.n_elecs && i < n_orbs; i++) {
      std::cout << eigenvalues[i] << "\n";
    }
  }
  Timer::checkpoint("compute natural orbitals");
}

void Optimization::rotate_and_rewrite_integrals() {
  rotate_integrals();
  rewrite_integrals();
}

void Optimization::rotate_integrals() {
  Timer::start("rotate integrals");
  new_integrals.allocate(n_orbs);
  IntegralsArray tmp_integrals;
  tmp_integrals.allocate(n_orbs);

// Two-body integrals
#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          tmp_integrals.get_2b(p, q, r, s) = integrals.get_2b(p, q, r, s);
        } // s
      }   // r
    }     // q
  }       // p

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, p) * tmp_integrals.get_2b(i, q, r, s);
          }
          new_integrals.get_2b(p, q, r, s) = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, q) * tmp_integrals.get_2b(p, i, r, s);
          }
          new_integrals.get_2b(p, q, r, s) = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, r) * tmp_integrals.get_2b(p, q, i, s);
          }
          new_integrals.get_2b(p, q, r, s) = new_val;
        } // s
      }   // r
    }     // q
  }       // p

  tmp_integrals = new_integrals;

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          double new_val = 0.;
          for (unsigned i = 0; i < n_orbs; i++) {
            new_val += rot(i, s) * tmp_integrals.get_2b(p, q, r, i);
          }
          new_integrals.get_2b(p, q, r, s) = new_val;
        } // s
      }   // r
    }     // q
  }       // p

// One-body integrals
#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      tmp_integrals.get_1b(p, q) = integrals.get_1b(p, q);
    }
  }

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i = 0; i < n_orbs; i++) {
        new_val += rot(i, p) * tmp_integrals.get_1b(i, q);
      }
      new_integrals.get_1b(p, q) = new_val;
    }
  }

  tmp_integrals = new_integrals;

#pragma omp parallel for collapse(2)
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      double new_val = 0.;
      for (unsigned i = 0; i < n_orbs; i++) {
        new_val += rot(i, q) * tmp_integrals.get_1b(p, i);
      }
      new_integrals.get_1b(p, q) = new_val;
    }
  }
  Timer::end();
}

void Optimization::rewrite_integrals() {
  // replace integrals with new_integrals
  Timer::start("rewrite integrals");

  integrals.integrals_2b.clear();
  integrals.integrals_1b.clear();

  unsigned p, q, r, s;
  const double* integrals_ptr = new_integrals.data_2b();
  for (p = 0; p < n_orbs; p++) {
    for (q = 0; q < n_orbs; q++) {
      for (r = 0; r < n_orbs; r++) {
        for (s = 0; s < n_orbs; s++) {
          integrals.integrals_2b.set(Integrals::combine4(p, q, r, s), *integrals_ptr,
                                        [&](double &a, const double &b) {
                                          if (std::abs(a) < std::abs(b))
                                            a = b;
                                        });
          integrals_ptr++;
        }
      }
    }
  }

  integrals_ptr = new_integrals.data_1b();
  for (p = 0; p < n_orbs; p++) {
    for (q = 0; q < n_orbs; q++) {
      integrals.integrals_1b.set(Integrals::combine2(p, q), *integrals_ptr,
                                    [&](double &a, const double &b) {
                                      if (std::abs(a) < std::abs(b))
                                        a = b;
                                    });
      integrals_ptr++;
    }
  }
  Timer::end();
}

void Optimization::get_optorb_rotation_matrix_from_newton() {
  rdm.get_2rdm(hamiltonian_matrix);
  rdm.get_1rdm_from_2rdm();
  std::vector<index_t> param_indices = parameter_indices();
  VectorXd grad = gradient(param_indices);
  MatrixXdR hess = hessian(param_indices);
  rdm.clear();
  size_t dim = param_indices.size();

  SelfAdjointEigenSolver<MatrixXd> es(hess);
  double Hoo_lowest = es.eigenvalues().minCoeff();
  if (Parallel::is_master())
    printf("Lowest eigenvalue of Hoo: %.3E\n", Hoo_lowest);

  if (Hoo_lowest < 0.) {
    const double diag_shift = - 1.5 * Hoo_lowest;
    hess += diag_shift * MatrixXd::Identity(dim, dim);
    if (Parallel::is_master())
      printf("Diagonal shift: %.3E.\n", diag_shift);
  }
  
  // rotation matrix
  // VectorXd new_param = hess.fullPivLu().solve(-1 * grad);
  VectorXd new_param = hess.householderQr().solve(-1 * grad);

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
        inner_prod += old_update(i) * new_update(i) * hess(i, i);
        new_norm += new_update(i) * new_update(i) * hess(i, i);
        old_norm += old_update(i) * old_update(i) * hess(i, i);
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
}

void Optimization::get_optorb_rotation_matrix_from_approximate_newton() {
  rdm.get_2rdm(hamiltonian_matrix);
  rdm.get_1rdm_from_2rdm();
  std::vector<index_t> param_indices = parameter_indices();
  VectorXd grad = gradient(param_indices);
  MatrixXd hess_diag = hessian_diagonal(param_indices);
  rdm.clear();
  size_t dim = param_indices.size();

  VectorXd new_param(dim);
  for (size_t i = 0; i < dim; i++) {
    hess_diag(i) = std::max(hess_diag(i), 1e-5);
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
      double momentum_proportion = Config::get<double>("optimization/momentum_proportion", 0.0);
      //old_update += new_update;
      old_update = momentum_proportion * (old_update / old_norm) 
      		   + (1.0 - momentum_proportion) * (new_update / step_size / new_norm);
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
}

void Optimization::get_optorb_rotation_matrix_from_grad_descent() {
  rdm.get_2rdm(hamiltonian_matrix);
  rdm.get_1rdm_from_2rdm();
  std::vector<index_t> param_indices = parameter_indices();

  VectorXd grad = gradient(param_indices);
  rdm.clear();

  VectorXd new_param = -0.01 * grad;
  fill_rot_matrix_with_parameters(new_param, param_indices);
}

void Optimization::get_optorb_rotation_matrix_from_amsgrad() {
  rdm.get_2rdm(hamiltonian_matrix);
  rdm.get_1rdm_from_2rdm();
  std::vector<index_t> param_indices = parameter_indices();
  unsigned dim = param_indices.size();

  static std::vector<std::vector<double>> history(2,
                                                  std::vector<double>(dim, 0.));

  VectorXd grad = gradient(param_indices);
  rdm.clear();

  double eps = 1e-8;
  double eta = Config::get<double>("optimization/parameters/eta", 0.01);
  double beta1 = Config::get<double>("optimization/parameters/beta1", 0.5);
  double beta2 = Config::get<double>("optimization/parameters/beta2", 0.5);

  if (Parallel::is_master()) {
    std::cout << "learning parameters: eta, beta1, beta2 = " << eta << ", "
              << beta1 << ", " << beta2 << "\n";
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

  fill_rot_matrix_with_parameters(new_param, param_indices);
}

void Optimization::generate_optorb_integrals_from_bfgs() {
  rdm.get_2rdm(hamiltonian_matrix);
  rdm.get_1rdm_from_2rdm();
  std::vector<index_t> param_indices = parameter_indices();
  unsigned dim = param_indices.size();

  static bool restart = true;
  static VectorXd grad_prev, update_prev;
  static MatrixXd hess;
  static double initial_update_norm;

  VectorXd grad = gradient(param_indices);

  if (restart) {
    VectorXd hess_diag = hessian_diagonal(param_indices);
    for (size_t i = 0; i < dim; i++) hess_diag[i] = std::max(hess_diag[i] / 4., 1e-5);
    hess = hess_diag.asDiagonal();
    grad_prev = VectorXd::Zero(dim);
    update_prev = VectorXd::Zero(dim);
  }
  rdm.clear();
  const VectorXd y = grad - grad_prev;
  const VectorXd hs = hess * update_prev;
  const double ys = y.dot(update_prev);
  const double shs = hs.dot(update_prev);
  if (Parallel::is_master()) printf("ys, shs = %.5f, %.5f\n", ys, shs);
  const bool update_hessian = ys > 1e-6 && abs(shs) > 1e-6;
  if (update_hessian) {
    hess += y * y.transpose() / ys - hs * hs.transpose() / shs;
  } else {
    if (Parallel::is_master()) printf("Skip updating Hessian\n");
  }

  /*
  Timer::start("Eigen diagonalization of Hoo");
  SelfAdjointEigenSolver<MatrixXd> es(hess);
  double Hoo_lowest = es.eigenvalues().minCoeff();
  std::cout<<"Hoo_lowest "<<Hoo_lowest<<std::endl;
  Timer::end();
  */

  Timer::start("Eigen linsolve of Hoo");
  VectorXd new_param = (hess + 1e-3 * MatrixXd::Identity(dim, dim)).householderQr().solve(-grad);
  Timer::end();

  double update_norm = new_param.norm();
  static double step_size_factor = 5.;
  if (restart) {
    initial_update_norm = new_param.norm();
    restart = false;
  } else {
    if (update_norm > step_size_factor * initial_update_norm) {
      if (Parallel::is_master()) printf("Applying step size control.\n");
      const double new_update_norm = step_size_factor * initial_update_norm;
      new_param /= update_norm / new_update_norm;
      update_norm = new_update_norm;
    }
    step_size_factor = std::max(0.25, step_size_factor * 0.97); // shrink max step size every iter
  }
  if (Parallel::is_master())
    printf("norm of orbital gradient: %.5f, norm of orbital update: %.5f.\n", grad.norm(),
           update_norm);
 
  grad_prev = grad;
  update_prev = new_param;

  fill_rot_matrix_with_parameters(new_param, param_indices);
}

std::vector<Optimization::index_t> Optimization::parameter_indices() const {
  // return vector of (row,col) indices of optimization parameters
  const std::vector<unsigned>& orb_sym = integrals.orb_sym;
  const std::vector<unsigned>& highest_occ_orb_in_irrep = integrals.highest_occ_orb_in_irrep;
  std::vector<index_t> indices;
  const bool enforce_active_space = Config::get<bool>("chem/active_space", false);
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = i + 1; j < n_orbs; j++) {
      if (orb_sym[i] == orb_sym[j]) {
        if (enforce_active_space 
		&& highest_occ_orb_in_irrep[orb_sym[i] - 1] < i 
		&& highest_occ_orb_in_irrep[orb_sym[j] - 1] < j)
          continue; // virtual-virtual rotations are redundant
        indices.push_back(std::make_pair(i, j));
      }
    }
  }
  if (Parallel::is_master()) {
    std::cout << "Number of optimization parameters: " << indices.size()
              << std::endl;
  }
  return indices;
}

void Optimization::fill_rot_matrix_with_parameters(
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

#pragma omp parallel for
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = 0; j < n_orbs; j++) {
      for (unsigned k = 0; k < n_orbs; k++) {
        double cos_tau, sinc_tau;
        if (std::abs(Tau2(k)) < 1e-10) {
          cos_tau = 1;
          sinc_tau = 1;
        } else {
          double tau = std::sqrt(-Tau2(k));
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

VectorXd Optimization::gradient(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) {
  unsigned n_param = param_indices.size();
  VectorXd grad(n_param);
  if (generalized_Fock_matrix.rows() * generalized_Fock_matrix.cols() != n_param * n_param) get_generalized_Fock();
#pragma omp parallel for
  for (unsigned i = 0; i < n_param; i++) {
    unsigned p = param_indices[i].first;
    unsigned q = param_indices[i].second;
    grad(i) = 2 * (generalized_Fock_matrix(p, q) - generalized_Fock_matrix(q, p));
  }
  Timer::checkpoint("compute gradient");
  return grad;
}

Optimization::MatrixXdR Optimization::hessian(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) {
  unsigned n_param = param_indices.size();
  MatrixXd hessian(n_param, n_param);
  if (generalized_Fock_matrix.rows() * generalized_Fock_matrix.cols() != n_param * n_param) get_generalized_Fock();
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned i = 0; i < n_param; i++) {
    for (unsigned j = 0; j <= i; j++) {
      unsigned p = param_indices[i].first;
      unsigned q = param_indices[i].second;
      unsigned r = param_indices[j].first;
      unsigned s = param_indices[j].second;
      if (i == j) {
        hessian(i, j) = hessian_part(p, q, r, s) - 2 * hessian_part(p, q, s, r) + hessian_part(q, p, s, r);      
      } else {
        hessian(i, j) = hessian_part(p, q, r, s) - hessian_part(p, q, s, r) -
                        hessian_part(q, p, r, s) + hessian_part(q, p, s, r);
        hessian(j, i) = hessian(i, j);
      }
    }
  }
  Timer::checkpoint("compute hessian");
  return hessian;
}

VectorXd Optimization::hessian_diagonal(
    const std::vector<std::pair<unsigned, unsigned>> &param_indices) {
  unsigned n_param = param_indices.size();
  VectorXd hessian_diagonal(n_param);
  if (generalized_Fock_matrix.rows() * generalized_Fock_matrix.cols() != n_param * n_param) get_generalized_Fock();
#pragma omp parallel for
  for (unsigned i = 0; i < n_param; i++) {
    unsigned p = param_indices[i].first;
    unsigned q = param_indices[i].second;
    hessian_diagonal(i) = hessian_part(p, q, p, q) - 2 * hessian_part(p, q, q, p) + hessian_part(q, p, q, p);
  }
  Timer::checkpoint("compute diagonal of hessian");
  return hessian_diagonal;
}

void Optimization::get_generalized_Fock() {
  // A properly resized generalized_Fock_matrix indicates it's been computed
  generalized_Fock_matrix.resize(n_orbs, n_orbs);

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (unsigned i = 0; i < n_orbs; i++) {
    for (unsigned j = 0; j < n_orbs; j++) {
      if (integrals.orb_sym[i] == integrals.orb_sym[j])
        generalized_Fock_matrix(i, j) = generalized_Fock_element(i, j);
      else
        generalized_Fock_matrix(i, j) = 0.;
    }
  }
}

double Optimization::generalized_Fock_element(const unsigned m, const unsigned n) const {
  // Helgaker (10.8.24)
  double elem = 0.;
  for (unsigned q = 0; q < n_orbs; q++) {
    elem += rdm.one_rdm_elem(m, q) * integrals.get_1b(n, q);
  }
  for (unsigned q = 0; q < n_orbs; q++) {
    for (unsigned r = 0; r < n_orbs; r++) {
      for (unsigned s = 0; s < n_orbs; s++) {
        elem +=
            rdm.two_rdm_elem(m, r, s, q) * integrals.get_2b(n, q, r, s);
      }
    }
  }
  return elem;
}

double Optimization::Y_matrix(const unsigned p, const unsigned q, const unsigned r,
                              const unsigned s) const {
  // Helgaker (10.8.50)
  double elem = 0.;
  for (unsigned m = 0; m < n_orbs; m++) {
    for (unsigned n = 0; n < n_orbs; n++) {
      elem +=
          (rdm.two_rdm_elem(p, r, n, m) + rdm.two_rdm_elem(p, n, r, m)) *
          integrals.get_2b(q, m, n, s);
      elem += rdm.two_rdm_elem(p, m, n, r) * integrals.get_2b(q, s, m, n);
    }
  }
  return elem;
}

double Optimization::hessian_part(const unsigned p, const unsigned q, const unsigned r,
                                  const unsigned s) const {
  // Helgaker (10.8.53) content in [...]
  double elem = 0.;
  elem += 2 * rdm.one_rdm_elem(p, r) * integrals.get_1b(q, s);
  if (q == s)
    elem -= (generalized_Fock_matrix(p, r) + generalized_Fock_matrix(r, p));
  elem += 2 * Y_matrix(p, q, r, s);
  return elem;
}
