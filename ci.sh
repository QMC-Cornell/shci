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

# Download Boost.
echo "Downloading Boost"
wget -O boost_1_66_0.tar.gz https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar xzf boost_1_66_0.tar.gz
mkdir -p $TOOLS_DIR/boost/include
mv boost_1_66_0/boost $TOOLS_DIR/boost/include/

if [ -f "$TOOLS_DIR/boost/lib/libboost_mpi.so" ] || [ -f "$TOOLS_DIR/boost/lib/libboost_mpi.dylib" ]; then
	echo "Found cached Boost"
else
	echo "Downloading Boost"
  mkdir -p downloads
  cd downloads
	wget -O boost_1_65_1.tar.bz2 https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2
	tar xjf boost_1_65_1.tar.bz2
	echo "Configuring and building Boost"
	cd boost_1_65_1
  mkdir -p $TOOLS_DIR/boost
  ./bootstrap.sh
  echo 'libraries =  --with-mpi --with-serialization ;' >> project-config.jam
  echo 'using mpi : mpic++ ;' >> project-config.jam
	echo 'using gcc : 6 ;' >> project-config.jam
	./b2 -j8 --prefix=$TOOLS_DIR/boost install
	echo "Completed"
	echo
	cd ../../
fi
export PATH=$TOOLS_DIR/boost/bin:$PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/boost/lib:$LD_LIBRARY_PATH

make -j
make test_mpi
