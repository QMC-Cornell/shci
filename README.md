# Arrow  

Semistochastic Heat Bath Configuration Interaction (SHCI) method.
A very fast selected configuration interaction plus perturbation theory method for obtaining almost exact Full Configuration Interaction (FCI) energies.  Such methods have been used for more than 50 years, but SHCI has innovations that we believe make it faster than all its competitors.
Arrow enables massively parallel computations using the SHCI method with MPI+OpenMP hybrid parallelization.

[![Build Status](https://travis-ci.com/QMC-Cornell/shci.svg?token=Gy8pVWtUBHq57qdtpAKN&branch=master)](https://travis-ci.com/QMC-Cornell/shci) 

## How to Run
Instructions for installing and running Arrow are documented in the [wiki](https://github.com/QMC-Cornell/shci/wiki).

## How to Contribute

Arrow is a research program rather than a fully tested catch-all software package.
The efficiency and correctness of edge cases, or input values that differ greatly from the default or published values are not guaranteed.
We welcome help with extending the capabilities of Arrow.  If interested, please contact Cyrus Umrigar <CyrusUmrigar@cornell.edu>, or a member of his research group.

## History and Authorship
`Arrow` implements algorithms developed in the Umrigar group at Cornell, with considerable input from Sandeep Sharma at Boulder, CO.  Arrow was conceived and implemented by Junhao Li, and greatly extended by Yuan Yao and Tyler Anderson.  Written in C++, it contains improved versions of SHCI algorithms first developed by Adam Holmes and extended by Matt Otten, implemented in the FORTRAN `sqmc` program.  `sqmc` was originally developed by Frank Petruzielo, Hitesh Changlani and Adam Holmes to make major improvements to the Full Configuration Interaction Quantum Monte Carlo (FCIQMC) method, so it has both SHCI and FCIQMC capabilities.  All authors, aside from Sharma, were at Cornell when they contributed.

## Citations
Any papers that use Arrow should cite the following 3 papers:

"Fast semistochastic heat-bath configuration interaction", Junhao Li, Matthew Otten, Adam A. Holmes, Sandeep Sharma, and C. J. Umrigar,  J. Chem. Phys., 149, 214110 (2018).

"Heat-bath configuration interaction: An efficient selected configuration interaction algorithm inspired by heat-bath sampling", Adam A. Holmes, Norm M. Tubman, and C. J. Umrigar, J. Chem. Theory Comput. 12, 3674 (2016).

"Semistochastic heat-bath configuration interaction method: selected configuration interaction with semistochastic perturbation theory", Sandeep Sharma, Adam A. Holmes, Guillaume Jeanmairet, Ali Alavi, and C. J. Umrigar, J. Chem. Theory Comput. 13, 1595 (2017).

For the orbital optimization solver in Arrow, please cite:

"Orbital Optimization in Selected Configuration Interaction Methods", Yuan Yao and C. J. Umrigar, J. Chem. Theory Comput. 2021, 17, 4183 (2021).

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

@article{YaoUmr-JCTC-21,
Author={Yuan Yao and C. J. Umrigar},
Title={Orbital Optimization in Selected Conguration Interaction Methods},
Journal={J. Chem. Theory Comput.},
Volume = {17},
Pages = {4183-4194},
year={2021}
}
```
