#!/usr/bin/env bash

set -e

### Clear the directory before running hipify, or else the old .prehip files will be used ###
sudo rm -rf aten/hip-src/

#### Create HIP aten folder ####
mkdir -p aten/hip-src

### Copy over the files ###
cp -R aten/src/ -T aten/hip-src/

cd aten/hip-src/

# Extract the aten CMakeLists file.
cp ../CMakeLists.txt.hip ../CMakeLists.txt

# Extract the THC (.hip) files.
cp THC/CMakeLists.txt.hip THC/CMakeLists.txt
cp THC/THCAllocator.c.hip THC/THCAllocator.c
cp THC/THCApply.cuh.hip THC/THCApply.cuh
cp THC/THCBlas.cu.hip THC/THCBlas.cu
cp THC/THCNumerics.cuh.hip THC/THCNumerics.cuh
cp THC/THCTensorRandom.cu.hip THC/THCTensorRandom.cu
cp THC/THCTensorRandom.h.hip THC/THCTensorRandom.h
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
find ATen/ -name "*.prehip" -type f -delete

# Due to an issue in HCC, change filename of CuDNN batch norm
mv ATen/native/cudnn/BatchNorm.cpp ATen/native/cudnn/BatchNormCuDNN.cpp

# Disable OpenMP in aten/hip-src/TH/generic/THTensorMath.c
sed -i 's/_OPENMP/_OPENMP_STUBBED/g' TH/generic/THTensorMath.c

# Sparse Hipify
declare -a arr=("THC/THCGeneral.h.in" "THC/THCGeneral.cpp" "THC/generic/THCStorage.c" "THC/THCTensorRandom.cuh")

for i in "${arr[@]}"
do
  # Hipify the source file in place.
  hipify-perl -i $i

  sed -i 's/cudaHostAllocator/hipHostAllocator/g' $i
  sed -i 's/cudaErrorInsufficientDriver/hipErrorInsufficientDriver/g' $i
  sed -i 's/curand.h/hiprng.h/g' $i
  sed -i 's/curand_kernel.h/hiprng_kernel.h/g' $i
  sed -i 's/curand_uniform/hiprng_uniform/g' $i
  sed -i 's/curand_uniform_double/hiprng_uniform_double/g' $i

  sed -i 's/cusparseStatus_t/hipsparseStatus_t/g' $i
  sed -i 's/cusparseHandle_t/hipsparseHandle_t/g' $i
  sed -i 's/cusparseCreate/hipsparseCreate/g' $i

  sed -i 's/CUSPARSE_STATUS_SUCCESS/HIPSPARSE_STATUS_SUCCESS/g' $i
  sed -i 's/CUSPARSE_STATUS_NOT_INITIALIZED/HIPSPARSE_STATUS_NOT_INITIALIZED/g' $i
  sed -i 's/CUSPARSE_STATUS_ALLOC_FAILED/HIPSPARSE_STATUS_ALLOC_FAILED/g' $i
  sed -i 's/CUSPARSE_STATUS_INVALID_VALUE/HIPSPARSE_STATUS_INVALID_VALUE/g' $i
  sed -i 's/CUSPARSE_STATUS_MAPPING_ERROR/HIPSPARSE_STATUS_MAPPING_ERROR/g' $i
  sed -i 's/CUSPARSE_STATUS_EXECUTION_FAILED/HIPSPARSE_STATUS_EXECUTION_FAILED/g' $i
  sed -i 's/CUSPARSE_STATUS_INTERNAL_ERROR/HIPSPARSE_STATUS_INTERNAL_ERROR/g' $i

  # Blas Hipify
  sed -i 's/cublasStatus_t/hipblasStatus_t/g' $i
  sed -i 's/cublasHandle_t/hipblasHandle_t/g' $i
  sed -i 's/cublasCreate/hipblasCreate/g' $i

  sed -i 's/CUBLAS_STATUS_SUCCESS/HIPBLAS_STATUS_SUCCESS/g' $i
  sed -i 's/CUBLAS_STATUS_NOT_INITIALIZED/HIPBLAS_STATUS_NOT_INITIALIZED/g' $i
  sed -i 's/CUBLAS_STATUS_ALLOC_FAILED/HIPBLAS_STATUS_ALLOC_FAILED/g' $i
  sed -i 's/CUBLAS_STATUS_INVALID_VALUE/HIPBLAS_STATUS_INVALID_VALUE/g' $i
  sed -i 's/CUBLAS_STATUS_MAPPING_ERROR/HIPBLAS_STATUS_MAPPING_ERROR/g' $i
  sed -i 's/CUBLAS_STATUS_EXECUTION_FAILED/HIPBLAS_STATUS_EXECUTION_FAILED/g' $i
  sed -i 's/CUBLAS_STATUS_INTERNAL_ERROR/HIPBLAS_STATUS_INTERNAL_ERROR/g' $i

  # Headers
  sed -i 's/cusparse.h/hipsparse.h/g' $i
  sed -i 's/cublas_v2.h/hipblas.h/g' $i
  sed -i 's/#include "cuda.h"//g' $i

  # Allocators
  sed -i 's/cudaHostAllocator/hipHostAllocator/g' $i
  sed -i 's/cudaUVAAllocator/hipUVAAllocator/g' $i
  sed -i 's/cudaDeviceAllocator/hipDeviceAllocator/g' $i

  sed -i 's/cublasDestroy/hipblasDestroy/g' $i
  sed -i 's/cusparseDestroy/hipsparseDestroy/g' $i

  # More Pairings
  sed -i 's/curandStateMtgp32/hiprngStateMtgp32/g' $i
  sed -i 's/#define MAX_NUM_BLOCKS 200/#define MAX_NUM_BLOCKS 64 /g' $i
  sed -i 's/curand_log_normal_double/hiprng_log_normal_double/g' $i
  sed -i 's/curand_log_normal/hiprng_log_normal/g' $i
  sed -i 's/assert/\/\/assert/g' $i # Disable asserts on device code
  sed -i 's/curand_uniform/hiprng_uniform /g' $i
done

# Swap the math functions from std::pow to powf
sed -i '/s/std::pow/powf/g' ATen/native/cuda/Embedding.cu

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

# Disable the loading of the CUDA runtime in torch/cuda/__init__.py
sed -i 's/_cudart = _load_cudart()/# _cudart = _load_cudart()/g' torch/cuda/__init__.py
sed -i 's/_cudart.cudaGetErrorName.restype = ctypes.c_char_p/# _cudart.cudaGetErrorName.restype = ctypes.c_char_p/g' torch/cuda/__init__.py
sed -i 's/_cudart.cudaGetErrorString.restype = ctypes.c_char_p/# _cudart.cudaGetErrorString.restype = ctypes.c_char_p/g' torch/cuda/__init__.py
sed -i 's/_lazy_call(_check_capability)/# _lazy_call(_check_capability)/g' torch/cuda/__init__.py
