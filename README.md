# SHCI Arrow  

` >>>====> ` Fast Semistochastic Heat Bath Configuration Interaction Solver for accurate quantum simulations.

[![Build Status](https://travis-ci.com/QMC-Cornell/shci.svg?token=Gy8pVWtUBHq57qdtpAKN&branch=master)](https://travis-ci.com/QMC-Cornell/shci) 

## Compilation
Make sure you have installed MPI.
```
git clone --recursive https://github.com/jl2922/shci
cd shci
make -j
```
`--recursive` will download all the dependencies from Github recursively.

## Example Run
An example carbon atom calculation inputs is provided with the code.
```
cp FCIDUMP.example FCIDUMP
cp config.json.example config.json
mpirun -n 1 ./shci
```
To run other systems, you will have to obtain an `FCIDUMP` file and modify the values in `config.json` accordingly.
Many software packages can generate `FCIDUMP`, such as [`PySCF`](https://github.com/sunqm/pyscf) and [`Molpro`](https://www.molpro.net/).

## Configurations
### General
* `n_up`, `n_dn` (required): number of up / down electrons.
* `system` (required): only support `chem` for now.
* `eps_vars` (required): an array of variational epsilons from big to small.
* `occs_up`: occupied orbitals for the starting up determinant, default to lowest ones.
* `occs_dn`: occupied orbitals for the starting dn determinant, default to lowest ones.
* `eps_vars_schedule`: :palm_tree: an array of additional variational epsilons run beforehand for a better selection of determinants, default to an empty array.
* `target_error`: :palm_tree: target error for stochastic perturbation, default to 1.0e-5.
* `var_only`: only run variation, default to false.
* `force_var`: run variation even if valid wavefunction files already exists, default to false.
* `var_sd`: :palm_tree: include all singles and doubles excitation, i.e. at least CISD, default to false.
* `get_pair_contrib`: :palm_tree: calculate occupied pair contribution, default to false.
* `eps_pt`: :palm_tree: perturbation epsilon, default to eps_var / 5000.
* `eps_pt_psto`: :palm_tree: pseudo stochastic perturbation epsilon, default to eps_var / 500.
* `eps_pt_dtm`: :palm_tree: deterministic perturbation epsilon, default to eps_var / 50.
* `max_pt_iteration`: :palm_tree: maximum stochastic perturbation iterations, default to 100.
* `n_batches_pt_sto`: :palm_tree: number of batches for stochastic perturbation, default to 16.
* `n_samples_pt_sto`: :palm_tree: number of samples for stochastic perturbation, default to choose based on available system memory.
* `random_seed`: for stochastic perturbation, default to the current timestamp.
* `time_sym`: :palm_tree: whether turns on time reversal symmetry, default to false.
* `s2`: :palm_tree: whether calculates s squared, default to false.
* `natorb`: whether generates natural orbitals FCIDUMP, default to false.
* `load_integrals_cache`: whether loads FCIDUMP information from integrals_cache, default to false.
* `get_1rdm_csv`, `get_2rdm_csv`: :seedling: whether calculates the density matrices, default to false. If it is true, the program outputs density matrices for the smallest `eps_var` in the `csv` format. For the two body density matrix, the `p q r s` columns represent `a+_p a+_q a_r a_s`.
* `get_green`: :seedling: whether calculates the green's function, default to false. If it is true, `w_green` gives the real part of the frequency and `n_green` gives the imaginary part. `advanced_green` determines whether calculating G- (true) or G+ (false), default to false. The Green's function matrix is returned in `csv` format.
* `hc_server_mode`: :seedling: operates as an H * c server, default to false. If it is true, the program serves as an RPC server for performing H * c matrix-vector multiplication after finishing the matrix reconstruction. It can work with any language that supports direct socket IO. A python client interface / demo is provided via `hc_client.py`. `hc_client.py` exposes a class called `HcClient`, which has three public methods: `getN` for getting the number of determinants, `getCoefs` for getting the coefficients array as a numpy array, and `Hc(arr)` which performs the matrix-vector multiplication on a numpy array of `dtype` either `np.float64` or `np.complex64` and returns the resulting numpy array of the same type. The `HcClient` accepts several optional construction options:
  - `nProcs`: default to 1, where the program uses all the cores on the master node. To run the `hc_server` across nodes, set `nProcs` to the number of nodes allocated to the job when creating the `HcClient` instance.
  - `runtimePath`: default to the current working directory. The `config.json` and `FCIDUMP` shall exist in the runtime path.
  - `shciPath`: default to the `shci` under the current working directory, probably needs to be changed to the actual path of the program.
  - `port`: default to 2018. Change it together with the value in `src/solver/hc_server.h` in case of a port conflict.
  - `verbose`: default to true.

:seedling: Experimental

:palm_tree: Experimental for the default values.

### Chemistry Block `chem`
* `point_group` (required): supports `C1`, `C2`, `Cs`, `Ci`, `C2v`, `C2h`, `Coov`, `D2`, `D2h`, and `Dooh`.
* `irreps`: an array of irreducible representations. If occupations are also given, they together determine the starting determinant, otherwise, the lowest orbitals are filled. `occs_up` and `occs_dn` when specified explicitly have priority over irreps.
* `irrep_occs_up` and `irrep_occs_dn`: occupation of each irreducible representation for up and down electrons respectively, the lowest orbitals satisfying which constitute the starting determinant. Ignored if `irreps` is not given.



## Citations
Li, Junhao, Matthew Otten, Adam A. Holmes, Sandeep Sharma, and Cyrus J. Umrigar. "Fast semistochastic heat-bath configuration interaction." The Journal of chemical physics 149, no. 21 (2018): 214110.

Holmes, Adam A., Norm M. Tubman, and C. J. Umrigar. "Heat-bath configuration interaction: An efficient selected configuration interaction algorithm inspired by heat-bath sampling." Journal of chemical theory and computation 12, no. 8 (2016): 3674-3680.

Sharma, Sandeep, Adam A. Holmes, Guillaume Jeanmairet, Ali Alavi, and Cyrus J. Umrigar. "Semistochastic heat-bath configuration interaction method: selected configuration interaction with semistochastic perturbation theory." Journal of chemical theory and computation 13, no. 4 (2017): 1595-1604.
