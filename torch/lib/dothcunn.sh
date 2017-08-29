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

echo $INSTALL_DIR
echo $C_FLAGS

#!/bin/bash
mkdir -p THCUNN/hip
mkdir -p THCUNN/hip/generic
cp THCUNN/*.h THCUNN/hip/
cp THCUNN/*.c THCUNN/hip/
cp THCUNN/*.cpp THCUNN/hip/
cp THCUNN/*.cu THCUNN/hip/
cp THCUNN/*.cuh THCUNN/hip/
cp THCUNN/generic/*.h THCUNN/hip/generic/
cp THCUNN/generic/*.c THCUNN/hip/generic/
cp THCUNN/generic/*.cu THCUNN/hip/generic/
cp THCUNN/CMakeLists.txt.hip THCUNN/hip/CMakeLists.txt
/root/wst_HIP/bin/hipconvertinplace-perl.sh THCUNN/hip/
/root/wst_HIP/bin/hipify-perl THCUNN/hip/THCUNNGeneral.h.in
find THCUNN/hip -name "*.prehip" -type f -delete

# Used to build an individual library, e.g. build TH
# function build() {
  # We create a build directory for the library, which will
  # contain the cmake output
  # mkdir -p build/$1
  mkdir -p build/THCUNN
  cd build/THCUNN
  BUILD_C_FLAGS=''
  # case $1 in
  case THCUNN in
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

