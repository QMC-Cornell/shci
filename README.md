# Arrow  

Semistochastic Heat Bath Configuration Interaction (SHCI) method.
A very fast selected configuration interaction plus perturbation theory method for obtaining almost exact Full Configuration Interaction (FCI) energies.  Such methods have been used for more than 50 years, but SHCI has innovations that we believe make it faster than all its competitors.

[![Build Status](https://travis-ci.com/QMC-Cornell/shci.svg?token=Gy8pVWtUBHq57qdtpAKN&branch=master)](https://travis-ci.com/QMC-Cornell/shci) 

## Compilation
Make sure you have installed MPI.
```
git clone --recursive https://github.com/QMC-Cornell/shci
cd shci
git checkout stable # optional
git submodule update --init --recursive
make -j
```
If using intel compiler, compile with `make -f Makefile.intel -j`.

## Example Run
Example inputs and outputs for carbon and chromium atoms and nitrogen molecule are in the examples directory.
All you need are config.json and FCIDUMP.  Run 1 MPI process per node and number of OpenMP threads/node = # cores/node.
```
mpirun -n 1 ../../shci > out
```
To run other systems, you will have to obtain an integrals file, `FCIDUMP`, and modify the values in `config.json` accordingly.
Many software packages can generate `FCIDUMP`, such as [`PySCF`](https://github.com/sunqm/pyscf) and [`Molpro`](https://www.molpro.net/).
If running a system with > 128 orbitals, edit shci/src/det/half_det.h and change #define N_CHUNKS 2 so that N_CHUNKS * 64 >= n_orb.

## How to contribute

Arrow is a research program rather than a fully tested catch-all software package.
The efficiency and correctness of edge cases, or input values that differ greatly from the default or published values are not guaranteed.
We welcome help with extending the capabilities of Arrow.  If interested, please contact Cyrus Umrigar <CyrusUmrigar@cornell.edu>, or a member of his research group.

## Input (config.json)

Only the most important input variables are listed below.

### General
* `n_up`, `n_dn` (required): number of up / down electrons.
* `system` (required): only support `chem` and `heg` for now.
* `occs_up`: occupied orbitals for the starting up determinant, default: lowest ones.
* `occs_dn`: occupied orbitals for the starting dn determinant, default: lowest ones.
* `eps_vars` (required): an array of variational epsilons from big to small.
* `eps_vars_schedule`: an array of additional variational epsilons run beforehand for a better selection of determinants, default: an empty array.
* `eps_pt_dtm`: :palm_tree: deterministic perturbation epsilon, default: `max(eps_var*eps_pt_dtm_ratio,2e-6)`.
* `eps_pt_psto`: :palm_tree: pseudo stochastic perturbation epsilon, default: `max(eps_var*eps_pt_psto_ratio,1e-7)`
* `eps_pt`: :palm_tree: perturbation epsilon, default: `eps_var*eps_pt_ratio`.
* `eps_pt_dtm_ratio`: :palm_tree: ratio eps_pt_dtm/eps_var, default: eps_var / 10.
* `eps_pt_psto_ratio`: :palm_tree: ratio eps_pt_psto/eps_var, default: eps_var / 100.
* `eps_pt_ratio`: :palm_tree: ratio eps_pt/eps_var, default: eps_var / 1000.
* `max_pt_iterations`: :palm_tree: maximum stochastic perturbation iterations, default: 100.
* `n_batches_pt_sto`: :palm_tree: number of batches for stochastic perturbation, default: 16.
* `n_samples_pt_sto`: :palm_tree: number of samples for stochastic perturbation, default: choose based on available system memory.
* `random_seed`: for stochastic perturbation, default: 347634253.
* `target_error`: target error for stochastic perturbation, default: 1.0e-5.
* `var_only`: run variation only, useful e.g. when optimizing orbs, default: false.
* `force_var`: run variation even if valid wavefunction files already exist, useful e.g. when optimizing orbs, default: false.
* `skip_var`: skip the extra read of the wavefunction when wavefunction files already exist, default: false.
* `var_sd`: :palm_tree: include all singles and doubles excitation, i.e. at least CISD, default: false.
* `get_pair_contrib`: :palm_tree: calculate occupied pair contribution, default: false.
* `time_sym`: :palm_tree: uses time-reversal symmetry, recommended for singlets, default: false.
* `s2`: :palm_tree: calculates s squared, default: false.
* `natorb`: generates natural orbitals FCIDUMP, default: false.
* `optorb`: generates optimized orbitals FCIDUMP, default: false.
* `second_rejection`: it uses 2nd criterion for choosing dets, useful when core excit allowed, default: false.
* `second_rejection_factor`: default: false.
* `load_integrals_cache`: loads FCIDUMP information from integrals_cache, default: false.
* `get_1rdm_csv`, `get_2rdm_csv`: :seedling: calculates the density matrices, default: false. If it is true, the program outputs density matrices for the smallest `eps_var` in the `csv` format. For the two body density matrix, the `p q r s` columns represent `a+_p a+_q a_r a_s`.
* `get_green`: :seedling: calculates the Green's function, default: false. If it is true, `w_green` is the real part of the frequency and `n_green` is the imaginary part. `advanced_green` selects G- (true) or G+ (false), default: false. The Green's function matrix is returned in `csv` format.
* `hc_server_mode`: :seedling: operates as an H * c server, default: false. If it is true, the program serves as an RPC server for performing H * c matrix-vector multiplication after finishing the matrix reconstruction. It can work with any language that supports direct socket IO. A python client interface / demo is provided via `hc_client.py`. `hc_client.py` exposes a class called `HcClient`, which has three public methods: `getN` for getting the number of determinants, `getCoefs` for getting the coefficients array as a numpy array, and `Hc(arr)` which performs the matrix-vector multiplication on a numpy array of `dtype` either `np.float64` or `np.complex64` and returns the resulting numpy array of the same type. The `HcClient` accepts several optional construction options:
  - `nProcs`: default: 1, where the program uses all the cores on the master node. To run the `hc_server` across nodes, set `nProcs` to the number of nodes allocated to the job when creating the `HcClient` instance.
  - `runtimePath`: default: the current working directory. The `config.json` and `FCIDUMP` shall exist in the runtime path.
  - `shciPath`: default: the `shci` under the current working directory, probably needs to be changed to the actual path of the program.
  - `port`: default: 2018. Change it together with the value in `src/solver/hc_server.h` in case of a port conflict.
  - `verbose`: default: true.

:seedling: Experimental

:palm_tree: Experimental, or, default values may not be optimal.

### Chemistry Block `chem`
* `point_group` (required): supports `C1`, `C2`, `Cs`, `Ci`, `C2v`, `C2h`, `Coov`, `D2`, `D2h`, and `Dooh`.
* `irreps`: an array of irreducible representations. If occupations are also given, they together determine the starting determinant, otherwise, the lowest orbitals are filled. `occs_up` and `occs_dn` when specified (outside chem block) have priority over irreps.
* `irrep_occs_up` and `irrep_occs_dn`: occupation of each irreducible representation for up and down electrons respectively in the starting determinant.  The lowest orbitals of each irrep. are filled.  Ignored if `irreps` is not given.


### Optimization Block `optimization`
* `natorb_iter`: number of natural orbital iterations, default: 1.
* `optorb_iter`: number of iterations of full orbital optimization, default: 20.
* `method`: currently supported optimization methods include `app_newton` (Newton's method with the Hessian approximated by its diagonal; default), `newton` (Newton's method with the entire Hessian calculated), `amsgrad`, and `grad_descent` (gradient descent).
* `rotation_matrix`: write out rotation matrix for each optimization iteration, default: false.
* `accelerate`: use overshooting in optimization, default: false. Specific to `method`: `app_newton` and `newton`.
* `parameters` block: optimization parameters specific to `method`: `amsgrad`. `eta`: default: 0.01; `beta1`: default: 0.5; `beta2`: default: 0.5. 

## History and Authorship
`Arrow` implements algorithms developed in the Umrigar group at Cornell, with considerable input from Sandeep Sharma at Boulder, CO.  Arrow was conceived and implemented by Junhao Li, and greatly extended by Yuan Yao and Tyler Anderson.  Written in C++, it contains improved versions of SHCI algorithms first developed by Adam Holmes and extended by Matt Otten, implemented in the FORTRAN `sqmc` program.  `sqmc` was originally developed by Frank Petruzielo, Hitesh Changlani and Adam Holmes to make major improvements to the Full Configuration Interaction Quantum Monte Carlo (FCIQMC) method, so it has both SHCI and FCIQMC capabilities.  All authors, aside from Sharma, were at Cornell when they contributed.

## Citations
Any papers that use Arrow should cite the following 3 papers:

"Fast semistochastic heat-bath configuration interaction", Junhao Li, Matthew Otten, Adam A. Holmes, Sandeep Sharma, and C. J. Umrigar,  J. Chem. Phys., 149, 214110 (2018).

"Heat-bath configuration interaction: An efficient selected configuration interaction algorithm inspired by heat-bath sampling", Adam A. Holmes, Norm M. Tubman, and C. J. Umrigar, J. Chem. Theory Comput. 12, 3674 (2016).

"Semistochastic heat-bath configuration interaction method: selected configuration interaction with semistochastic perturbation theory", Sandeep Sharma, Adam A. Holmes, Guillaume Jeanmairet, Ali Alavi, and C. J. Umrigar, J. Chem. Theory Comput. 13, 1595 (2017).

In bibfile format:
```
@article{LiOttHolShaUmr-JCP-18,
Author = {Junhao Li and  Matthew Otten and Adam A. Holmes and Sandeep Sharma and C. J. Umrigar},
Title = {Fast Semistochastic Heat-Bath Configuration Interaction},
Journal = {J. Chem. Phys.},
Year = {2018},
Volume = {148},
Pages = {214110}
}

@article{ShaHolJeaAlaUmr-JCTC-17,
Author = {Sandeep Sharma and Adam A. Holmes and Guillaume Jeanmairet and Ali Alavi and C. J. Umrigar},
Title = {Semistochastic Heat-Bath Configuration Interaction Method: Selected
   Configuration Interaction with Semistochastic Perturbation Theory},
Journal = {J. Chem. Theory Comput.},
Year = {2017},
Volume = {13},
Pages = {1595-1604},
DOI = {10.1021/acs.jctc.6b01028},
}

@article{HolTubUmr-JCTC-16,
Author = {Adam A. Holmes and Norm M. Tubman and C. J. Umrigar},
Title = {Heat-bath Configuration Interaction: An efficient selected CI algorithm inspired by heat-bath sampling},
Journal = {J. Chem. Theory Comput.},
Volume = {12},
Pages = {3674-3680},
Year = {2016}}
}
```
