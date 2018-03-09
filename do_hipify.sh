#!/usr/bin/env bash

set -e

### Clear the directory before running hipify, or else the old .prehip files will be used ###
sudo rm -rf aten/hip-src/

#### Create HIP aten folder ####
mkdir -p aten/hip-src

### Copy over the files ###
cp -r aten/src/* aten/hip-src/

cd aten/hip-src/

# Extract the aten CMakeLists file.
cp ../CMakeLists.txt.hip ../CMakeLists.txt

# Extract the THC (.hip) files.
cp THC/CMakeLists.txt.hip THC/CMakeLists.txt
cp THC/THCAllocator.c.hip THC/THCAllocator.c
cp THC/THCApply.cuh.hip THC/THCApply.cuh
cp THC/THCBlas.cu.hip THC/THCBlas.cu
cp THC/THCGeneral.cpp.hip THC/THCGeneral.cpp
cp THC/THCGeneral.h.in.hip THC/THCGeneral.h.in
cp THC/THCNumerics.cuh.hip THC/THCNumerics.cuh
cp THC/THCTensorRandom.cu.hip THC/THCTensorRandom.cu
cp THC/THCTensorRandom.cuh.hip THC/THCTensorRandom.cuh
cp THC/THCTensorRandom.h.hip THC/THCTensorRandom.h
cp THC/generic/THCStorage.c.hip THC/generic/THCStorage.c
cp THC/generic/THCTensorRandom.cu.hip THC/generic/THCTensorRandom.cu

# Run hipify script in place
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THC/
/opt/rocm/hip/bin/hipify-perl THC/THCGeneral.h.in
find THC/ -name "*.prehip" -type f -delete

# Extract the THCUNN (.hip) files.
cp THCUNN/CMakeLists.txt.hip THCUNN/CMakeLists.txt
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THCUNN/
find THCUNN/ -name "*.prehip" -type f -delete

# Extract the THCS (.hip) files.
cp THCS/CMakeLists.txt.hip THCS/CMakeLists.txt
/opt/rocm/hip/bin/hipconvertinplace-perl.sh THCS/
find THCS/ -name "*.prehip" -type f -delete

# Extract the ATen files.
cp ATen/CMakeLists.txt.hip ATen/CMakeLists.txt
/opt/rocm/hip/bin/hipconvertinplace-perl.sh ATen/
#/opt/rocm/hip/bin/hipconvertinplace-perl.sh ATen/cuda/*.cu
#/opt/rocm/hip/bin/hipconvertinplace-perl.sh ATen/cuda/*.cuh
#/opt/rocm/hip/bin/hipconvertinplace-perl.sh ATen/native/cuda/*.cu
cp ../src/ATen/native/cuda/Distributions.cu ATen/native/cuda/Distributions.cu
sed -i 's/cudaHostAllocator/hipHostAllocator/g' ATen/PinnedMemoryAllocator.cpp
sed -i 's/cudaErrorInsufficientDriver/hipErrorInsufficientDriver/g' ATen/Context.cpp
sed -i 's/curand.h/hiprng.h/g' ATen/native/cuda/*
sed -i 's/curand_kernel.h/hiprng_kernel.h/g' ATen/native/cuda/*
sed -i 's/curand_uniform/hiprng_uniform/g' THC/generic/THCTensorRandom.cu
sed -i 's/curand_uniform_double/hiprng_uniform_double/g' THC/generic/THCTensorRandom.cu

find ATen/cuda/ -name "*.prehip" -type f -delete

# Disable OpenMP in aten/hip-src/TH/generic/THTensorMath.c
sed -i 's/_OPENMP/_OPENMP_STUBBED/g' TH/generic/THTensorMath.c

# Make link directories
mkdir -p HIP
cd HIP
if [ ! -L "TH" ]; then
    ln -s ../TH TH
fi
if [ ! -L "THS" ]; then
    ln -s ../THS THS
fi
if [ ! -L "THNN" ]; then
    ln -s ../THNN THNN
fi
if [ ! -L "THC" ]; then
    ln -s ../THC THC
fi
if [ ! -L "THCS" ]; then
    ln -s ../THCS THCS
fi
if [ ! -L "THCUNN" ]; then
    ln -s ../THCUNN THCUNN
fi
if [ ! -L "THD" ]; then
    ln -s ../THD THD
fi
if [ ! -L "THPP" ]; then
    ln -s ../THPP THPP
fi
if [ ! -L "ATen" ]; then
    ln -s ../ATen ATen
fi
cd ../../../.
