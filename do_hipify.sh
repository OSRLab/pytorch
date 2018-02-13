#!/usr/bin/env bash

set -e

cd aten/src/

# THC
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
cp THC/THCGeneral.cpp.hip THC/hip/THCGeneral.cpp
cp THC/THCAllocator.c.hip THC/hip/THCAllocator.c
cp THC/generic/THCStorage.c.hip THC/hip/generic/THCStorage.c
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THC/hip/
/opt/rocm/hip/bin/hipify-perl THC/hip/THCGeneral.h.in
find THC/hip -name "*.prehip" -type f -delete

# THCUNN
mkdir -p THCUNN/hip
mkdir -p THCUNN/hip/generic
cp THCUNN/*.h THCUNN/hip/
cp THCUNN/*.cu THCUNN/hip/
cp THCUNN/*.cuh THCUNN/hip/
cp THCUNN/generic/*.h THCUNN/hip/generic/
cp THCUNN/generic/*.cu THCUNN/hip/generic/
cp THCUNN/CMakeLists.txt.hip THCUNN/hip/CMakeLists.txt
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THCUNN/hip/
find THCUNN/hip -name "*.prehip" -type f -delete

# THCS
mkdir -p THCS/hip
mkdir -p THCS/hip/generic
cp THCS/*.h THCS/hip/
# cp THCS/*.c THCS/hip/
cp THCS/*.cu THCS/hip/
cp THCS/generic/*.h THCS/hip/generic/
# cp THCS/generic/*.c THCS/hip/generic/
cp THCS/generic/*.cu THCS/hip/generic/
cp THCS/CMakeLists.txt.hip THCS/hip/CMakeLists.txt
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THCS/hip/
find THCS/hip -name "*.prehip" -type f -delete

# Make link directories
mkdir -p HIP
cd HIP
if [ ! -L "THC" ]; then
    ln -s ../THC/hip THC
fi
if [ ! -L "THCS" ]; then
    ln -s ../THCS/hip THCS
fi
if [ ! -L "THCUNN" ]; then
    ln -s ../THCUNN/hip THCUNN
fi
if [ ! -L "THD" ]; then
    ln -s ../THD THD
fi
if [ ! -L "THPP" ]; then
    ln -s ../THPP THPP
fi
if [ ! -L "THS" ]; then
    ln -s ../THS THS
fi
if [ ! -L "ATen" ]; then
    ln -s ../ATen ATen
fi
cd ../../../.
