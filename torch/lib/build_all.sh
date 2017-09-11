#!/usr/bin/env bash

# Shell script used to build the torch/lib/* dependencies prior to
# linking the libraries and passing the headers to the Python extension
# compilation stage. This file is used from setup.py, but can also be
# called standalone to compile the libraries outside of the overall PyTorch
# build process.

set -e

# Options for building only a subset of the libraries
WITH_CUDA=0
WITH_NCCL=0
WITH_ROCM=0
WITH_DISTRIBUTED=0
for arg in "$@"; do
    if [[ "$arg" == "--with-cuda" ]]; then
        WITH_CUDA=1
        WITH_ROCM=0
    elif [[ "$arg" == "--with-nccl" ]]; then
        WITH_NCCL=1
    elif [[ "$arg" == "--with-distributed" ]]; then
        WITH_DISTRIBUTED=1
    elif [[ "$arg" == "--with-rocm" ]]; then
        WITH_ROCM=1
        WITH_CUDA=0
        WITH_NCCL=0
        WITH_DISTRIBUTED=0
    else
        echo "Unknown argument: $arg"
    fi
done

cd "$(dirname "$0")/../.."
BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR="$(pwd)/tmp_install"
C_FLAGS=" -DTH_INDEX_BASE=0 -I$INSTALL_DIR/include \
  -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC \
  -I$INSTALL_DIR/include/THS -I$INSTALL_DIR/include/THCS \
  -I$INSTALL_DIR/include/THPP -I$INSTALL_DIR/include/THNN \
  -I$INSTALL_DIR/include/THCUNN"
LDFLAGS="-L$INSTALL_DIR/lib "
LD_POSTFIX=".so.1"
LD_POSTFIX_UNVERSIONED=".so"
if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
    LD_POSTFIX=".1.dylib"
    LD_POSTFIX_UNVERSIONED=".dylib"
else
    LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
fi

# Used to build an individual library, e.g. build TH
function build() {
  # We create a build directory for the library, which will
  # contain the cmake output
  mkdir -p build/$1
  cd build/$1
  BUILD_C_FLAGS=''
  case $1 in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
              -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
              -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
              -DCUDA_NVCC_FLAGS="$C_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
              -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
              -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
              -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
              -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
              -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
              -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
              -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
              -DTH_SO_VERSION=1 \
              -DTHC_SO_VERSION=1 \
              -DTHNN_SO_VERSION=1 \
              -DTHCUNN_SO_VERSION=1 \
              -DTHD_SO_VERSION=1 \
              -DNO_CUDA=$((1-$WITH_CUDA)) \
              -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release) \
              $2
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  local lib_prefix=$INSTALL_DIR/lib/lib$1
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi

  if [[ $(uname) == 'Darwin' ]]; then
    cd tmp_install/lib
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    cd ../..
  fi
}
function build_rocm_THC() {
  mkdir -p THC/hip
  mkdir -p THC/hip/generic
  cp THC/*.h THC/hip/
  cp THC/*.c THC/hip/
  cp THC/*.cpp THC/hip/
  cp THC/*.cu THC/hip/
  cp THC/*.cuh THC/hip/
  cp THC/generic/*.h THC/hip/generic/
  cp THC/generic/*.c THC/hip/generic/
  cp THC/generic/*.cu THC/hip/generic/
  cp THC/CMakeLists.txt.hip THC/hip/CMakeLists.txt
  cp THC/THCGeneral.h.in.hip THC/hip/THCGeneral.h.in
  cp THC/THCBlas.cu.hip THC/hip/THCBlas.cu
  cp THC/THCApply.cuh.hip THC/hip/THCApply.cuh
  cp THC/THCTensorRandom.cu.hip THC/hip/THCTensorRandom.cu
  cp THC/THCTensorRandom.cuh.hip THC/hip/THCTensorRandom.cuh
  cp THC/THCTensorRandom.h.hip THC/hip/THCTensorRandom.h
  cp THC/THCNumerics.cuh.hip THC/hip/THCNumerics.cuh
  cp THC/generic/THCTensorRandom.cu.hip THC/hip/generic/THCTensorRandom.cu
  cp THC/THCGeneral.cc.hip THC/hip/THCGeneral.cc
  cp THC/THCAllocator.cc.hip THC/hip/THCAllocator.cc
  cp THC/generic/THCStorage.c.hip THC/hip/generic/THCStorage.c
  /root/wst_HIP/bin/hipconvertinplace-perl.sh THC/hip/
  /root/wst_HIP/bin/hipify-perl THC/hip/THCGeneral.h.in
  find THC/hip -name "*.prehip" -type f -delete

  mkdir -p build/THC
  cd build/THC
  BUILD_C_FLAGS=''
  # case $1 in
  case THC in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  cmake ../../THC/hip -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake" \
               -DTorch_FOUND="1" \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
               -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
               -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
               -DCUDA_NVCC_FLAGS="$C_FLAGS" \
               -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
               -DTH_LIB_PATH="$INSTALL_DIR/lib" \
               -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
               -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
               -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
               -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
               -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
               -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
               -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
               -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
               -DTH_SO_VERSION=1 \
               -DTHC_SO_VERSION=1 \
               -DTHNN_SO_VERSION=1 \
               -DTHCUNN_SO_VERSION=1 \
               -DTHD_SO_VERSION=1 \
               -DNO_CUDA=0 \
               -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release)
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  local lib_prefix=$INSTALL_DIR/lib/libTHC
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi
}
function build_rocm_THCUNN() {
  mkdir -p THCUNN/hip
  mkdir -p THCUNN/hip/generic
  cp THCUNN/*.h THCUNN/hip/
  cp THCUNN/*.cu THCUNN/hip/
  cp THCUNN/*.cuh THCUNN/hip/
  cp THCUNN/generic/*.h THCUNN/hip/generic/
  cp THCUNN/generic/*.cu THCUNN/hip/generic/
  cp THCUNN/CMakeLists.txt.hip THCUNN/hip/CMakeLists.txt
  /root/wst_HIP/bin/hipconvertinplace-perl.sh THCUNN/hip/
  find THCUNN/hip -name "*.prehip" -type f -delete

  # We create a build directory for the library, which will
  # contain the cmake output
  mkdir -p build/THCUNN
  cd build/THCUNN
  BUILD_C_FLAGS=''
  case THCUNN in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  cmake ../../THCUNN/hip -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake" \
               -DTorch_FOUND="1" \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
               -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
               -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
               -DCUDA_NVCC_FLAGS="$C_FLAGS" \
               -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
               -DTH_LIB_PATH="$INSTALL_DIR/lib" \
               -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
               -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
               -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
               -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
               -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
               -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
               -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
               -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
               -DTH_SO_VERSION=1 \
               -DTHC_SO_VERSION=1 \
               -DTHNN_SO_VERSION=1 \
               -DTHCUNN_SO_VERSION=1 \
               -DTHD_SO_VERSION=1 \
               -DNO_CUDA=0 \
               -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release)
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  local lib_prefix=$INSTALL_DIR/lib/libTHC
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi
}
function build_rocm_THCS() {
  mkdir -p THCS/hip
  mkdir -p THCS/hip/generic
  cp THCS/*.h THCS/hip/
  cp THCS/*.c THCS/hip/
  cp THCS/*.cu THCS/hip/
  cp THCS/generic/*.h THCS/hip/generic/
  cp THCS/generic/*.c THCS/hip/generic/
  cp THCS/generic/*.cu THCS/hip/generic/
  cp THCS/CMakeLists.txt.hip THCS/hip/CMakeLists.txt
  /root/wst_HIP/bin/hipconvertinplace-perl.sh THCS/hip/
  find THCS/hip -name "*.prehip" -type f -delete

  # We create a build directory for the library, which will
  # contain the cmake output
  mkdir -p build/THCS
  cd build/THCS
  BUILD_C_FLAGS=''
  # case $1 in
  case THCS in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  cmake ../../THCS/hip -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake" \
               -DTorch_FOUND="1" \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
               -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
               -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
               -DCUDA_NVCC_FLAGS="$C_FLAGS" \
               -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
               -DTH_LIB_PATH="$INSTALL_DIR/lib" \
               -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
               -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
               -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
               -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
               -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
               -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
               -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
               -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
               -DTH_SO_VERSION=1 \
               -DTHC_SO_VERSION=1 \
               -DTHNN_SO_VERSION=1 \
               -DTHCUNN_SO_VERSION=1 \
               -DTHD_SO_VERSION=1 \
               -DNO_CUDA=0 \
               -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release)
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  # local lib_prefix=$INSTALL_DIR/lib/lib$1
  local lib_prefix=$INSTALL_DIR/lib/libTHC
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi

  if [[ $(uname) == 'Darwin' ]]; then
    cd tmp_install/lib
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    cd ../..
  fi
}
function build_nccl() {
   mkdir -p build/nccl
   cd build/nccl
   cmake ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
               -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS"
   make install
   cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so.1"
   if [ ! -f "${INSTALL_DIR}/lib/libnccl.so" ]; then
     ln -s "${INSTALL_DIR}/lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so"
   fi
   cd ../..
}

# In the torch/lib directory, create an installation directory
mkdir -p tmp_install

# We need to build the CPU libraries first, because they are used
# in the CUDA libraries
build TH
build THS
build THNN

CPP_FLAGS=" -std=c++11 "
if [[ $WITH_ROCM -eq 1 ]]; then
    build_rocm_THC
    build_rocm_THCS
    build_rocm_THCUNN
fi
if [[ $WITH_CUDA -eq 1 ]]; then
    build THC
    build THCS
    build THCUNN
fi
if [[ $WITH_NCCL -eq 1 ]]; then
    build_nccl
fi

# THPP has dependencies on both CPU and CUDA, so build it
# after those libraries have been completed
build THPP

# The shared memory manager depends on TH
build libshm
build ATen

# THD, gloo have dependencies on Torch, CUDA, NCCL etc.
if [[ $WITH_DISTRIBUTED -eq 1 ]]; then
    if [ "$(uname)" == "Linux" ]; then
        if [ -d "gloo" ]; then
            GLOO_FLAGS=""
            if [[ $WITH_CUDA -eq 1 ]]; then
                GLOO_FLAGS="-DUSE_CUDA=1 -DNCCL_ROOT_DIR=$INSTALL_DIR"
            fi
            build gloo "$GLOO_FLAGS"
        fi
    fi
    build THD
fi

# If all the builds succeed we copy the libraries, headers,
# binaries to torch/lib
cp $INSTALL_DIR/lib/* .
cp THNN/generic/THNN.h .
cp THCUNN/generic/THCUNN.h .
cp -r $INSTALL_DIR/include .
cp $INSTALL_DIR/bin/* .

# this is for binary builds
if [[ $PYTORCH_BINARY_BUILD && $PYTORCH_SO_DEPS ]]
then
    echo "Copying over dependency libraries $PYTORCH_SO_DEPS"
    # copy over dependency libraries into the current dir
    cp $PYTORCH_SO_DEPS .
fi
