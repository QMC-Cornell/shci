# SHCI

[![Build Status](https://travis-ci.org/jl2922/shci.svg?branch=master)](https://travis-ci.org/jl2922/shci)

## Compilation
Make sure you have installed MPI.
```
git clone --recursive https://github.com/jl2922/shci
cd shci
make -j
```

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
* `eps_vars_schedule`: an array of additional variational epsilons run beforehand for a better selection of determinants, default to an empty array.
* `target_error`: target error for stochastic perturbation, default to 1.0e-5.
* `var_only`: only run variation, default to false.
* `for_var`: run variation even if valid wavefunction files already exists, default to false.
* `eps_pt`: perturbation epsilon, default to eps_var / 5000.
* `eps_pt_psto`: pseudo stochastic perturbation epsilon, default to eps_var / 500.
* `eps_pt_dtm`: deterministic perturbation epsilon, default to eps_var / 50.
* `max_pt_iteration`: maximum stochastic perturbation iterations, default to 100.
* `n_batches_pt_sto`: number of batches for stochastic perturbation, default to 16.
* `n_samples_pt_sto`: number of samples for stochastic perturbation, default to choose based on available system memory.
* `random_seed`: for stochastic perturbation, default to the current timestamp.
* `time_sym`: whether turns on time reversal symmetry, default to false.
* `s2`: whether calculates s squared, default to false.
* `natorb`: whether generates natural orbitals FCIDUMP, default to false.
* `load_integrals_cache`: whether loads FCIDUMP information from integrals_cache, default to false.

### Chemistry Block `chem`
* `point_group`: supports `dih`, `d2h`, `c2v` or `none`, default to `none`.
* `irreps`: an array of irreducible representations. If occupations are also given, they together determine the starting determinant, otherwise, the lowest orbitals are filled.
* `irrep_occs_up` and `irrep_occs_dn`: occupation of each irreducible representation for up and down electrons respectively, the lowest orbitals satisfying which constitute the starting determinant. Ignored if `irreps` is not given.

### Citation
TBA
