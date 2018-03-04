#!/bin/bash
# Install dependencies and run tests.

test -n $CC && unset CC
test -n $CXX && unset CXX

set -x

# Install or Load OpenMPI.
if [ -f "$TOOLS_DIR/openmpi/bin/mpic++" ] && [ -f "$TOOLS_DIR/openmpi/bin/mpic++" ]; then
  echo "Found cached OpenMPI"
else
  echo "Installing OpenMPI"
  mkdir -p downloads
  cd downloads
  wget -O openmpi-3.0.0.tar.bz2 https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2
  tar xjf openmpi-3.0.0.tar.bz2
  cd openmpi-3.0.0
  mkdir -p $TOOLS_DIR/openmpi
  ./configure --prefix=$TOOLS_DIR/openmpi CC=$C_COMPILER CXX=$CXX_COMPILER
  make -j 8
  make install
  cd ../../
fi
export PATH=$TOOLS_DIR/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/openmpi/lib:$LD_LIBRARY_PATH

# Download Boost.
echo "Downloading Boost"
wget -O boost_1_66_0.tar.gz https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar xzf boost_1_66_0.tar.gz
mkdir -p $TOOLS_DIR/boost/include
mv boost_1_66_0/boost $TOOLS_DIR/boost/include/

cp ci.mk local.mk
make -j
make test_mpi
