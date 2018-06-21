# SHCI

[![Build Status](https://travis-ci.org/jl2922/shci.svg?branch=master)](https://travis-ci.org/jl2922/shci)

## Compilation
Make sure you have installed MPI.
```
git clone https://github.com/jl2922/shci
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
Many software packages can generate `FCIDUMP`, such as `PySCF` and `Molpro`.
