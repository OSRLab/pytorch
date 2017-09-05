#pragma once

// RAII structs to set CUDA device

#include <string>
#include <stdexcept>

#include <ATen/ATen.h>

#ifdef WITH_CUDA
  #if defined(__HIP_PLATFORM_HCC__)
    #include <hip/hip_runtime.h>
  #else
    #include <cuda.h>
    #include <cuda_runtime.h>
  #endif
#endif

struct AutoGPU {
  explicit AutoGPU(int device=-1) {
    setDevice(device);
  }

  explicit AutoGPU(const at::Tensor& t) {
    setDevice(t.type().is_cuda() ? (int) t.get_device() : -1);
  }

  explicit AutoGPU(at::TensorList &tl) {
    if (tl.size() > 0) {
      auto& t = tl[0];
      setDevice(t.type().is_cuda() ? t.get_device() : -1);
    }
  }

  ~AutoGPU() {
#ifdef WITH_CUDA
    if (original_device != -1) {
#if defined(__HIP_PLATFORM_HCC__)
      hipSetDevice(original_device);
#else
      cudaSetDevice(original_device);
#endif
    }
#endif
  }

  inline void setDevice(int device) {
#ifdef WITH_CUDA
    if (device == -1) {
      return;
    }
    if (original_device == -1) {
#if defined(__HIP_PLATFORM_HCC__)
      cudaCheck(hipGetDevice(&original_device));
#else
      cudaCheck(cudaGetDevice(&original_device));
#endif
      if (device != original_device) {
#if defined(__HIP_PLATFORM_HCC__)
        cudaCheck(hipSetDevice(device));
#else
        cudaCheck(cudaSetDevice(device));
#endif
      }
    } else {
#if defined(__HIP_PLATFORM_HCC__)
      cudaCheck(hipSetDevice(device));
#else
      cudaCheck(cudaSetDevice(device));
#endif
    }
#endif
  }

  int original_device = -1;

private:
#ifdef WITH_CUDA
#if defined(__HIP_PLATFORM_HCC__)
  static void cudaCheck(hipError_t err) {
    if (err != hipSuccess) {
      std::string msg = "CUDA error (";
      msg += std::to_string(err);
      msg += "): ";
      msg += hipGetErrorString(err);
      throw std::runtime_error(msg);
    }
  }
#else
  static void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
      std::string msg = "CUDA error (";
      msg += std::to_string(err);
      msg += "): ";
      msg += cudaGetErrorString(err);
      throw std::runtime_error(msg);
    }
  }
#endif
#endif
};
