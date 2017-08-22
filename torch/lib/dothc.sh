#!/bin/bash
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

# Used to build an individual library, e.g. build TH
# function build() {
  # We create a build directory for the library, which will
  # contain the cmake output
  # mkdir -p build/$1
  mkdir -p build/THC
  cd build/THC
  BUILD_C_FLAGS=''
  # case $1 in
  case THC in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  # cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
  # cmake ../../THC/hip -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake" \
  #             -DTorch_FOUND="1" \
  #             -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  #             -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
  #             -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
  #             -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
  #             -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
  #             -DCUDA_NVCC_FLAGS="$C_FLAGS" \
  #             -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
  #             -DTH_LIB_PATH="$INSTALL_DIR/lib" \
  #             -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
  #             -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
  #             -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
  #             -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
  #             -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
  #             -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
  #             -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
  #             -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
  #             -DTH_SO_VERSION=1 \
  #             -DTHC_SO_VERSION=1 \
  #             -DTHNN_SO_VERSION=1 \
  #             -DTHCUNN_SO_VERSION=1 \
  #             -DTHD_SO_VERSION=1 \
  #             -DNO_CUDA=$((1-$WITH_CUDA)) \
  #             -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release)
  cmake -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake" ../../THC/hip
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
               ../../THC/hip
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
# }

