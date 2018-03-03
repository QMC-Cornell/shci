#!/bin/bash
# Install dependencies and run tests.

test -n $CC && unset CC
test -n $CXX && unset CXX

set -x

# Install or Load OpenMPI.
if [ -f "$TOOLS_DIR/openmpi/bin/mpic++" ] && [ -f "$TOOLS_DIR/openmpi/bin/mpic++" ]; then
  echo "Found cached OpenMPI"
else
  echo "Downloading OpenMPI Source"
  mkdir -p downloads
  cd downloads
  wget -O openmpi-3.0.0.tar.bz2 https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2
  tar xjf openmpi-3.0.0.tar.bz2
  echo "Configuring and building OpenMPI"
  cd openmpi-3.0.0
  mkdir -p $TOOLS_DIR/openmpi
  ./configure --prefix=$TOOLS_DIR/openmpi
  make -j 8
  make install
  echo "Completed"
  echo
  cd ../../
fi
export PATH=$TOOLS_DIR/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/openmpi/lib:$LD_LIBRARY_PATH

make -j
make test_mpi
