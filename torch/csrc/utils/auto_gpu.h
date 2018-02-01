#pragma once

// RAII structs to set CUDA device

#include <string>
#include <stdexcept>

#include <ATen/ATen.h>

#ifdef WITH_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_HCC__)
  #include <hip/hip_runtime.h>
#endif

struct AutoGPU {
  explicit AutoGPU(int device=-1) {
    setDevice(device);
  }

  explicit AutoGPU(const at::Tensor& t) {
    setDevice(t.type().isCuda() ? t.get_device() : -1);
  }

  explicit AutoGPU(at::TensorList &tl) {
    if (tl.size() > 0) {
      auto& t = tl[0];
      setDevice(t.type().isCuda() ? t.get_device() : -1);
    }
  }

  ~AutoGPU() {
    if (original_device != -1) {
#ifdef WITH_CUDA
      cudaSetDevice(original_device);
#elif defined(__HIP_PLATFORM_HCC__)
      hipSetDevice(original_device);
#else
#endif
    }
  }

  inline void setDevice(int device) {
    if (device == -1) {
      return;
    }

    if (original_device == -1) {
#ifdef WITH_CUDA
      cudaCheck(cudaGetDevice(&original_device));
#elif defined(__HIP_PLATFORM_HCC__)
      cudaCheck(hipGetDevice(&original_device));
#else
#endif
      if (device != original_device) {
#ifdef WITH_CUDA
        cudaCheck(cudaSetDevice(device));
#elif defined(__HIP_PLATFORM_HCC__)
        cudaCheck(hipSetDevice(device));
#else
#endif
      }
    } else {
#ifdef WITH_CUDA
      cudaCheck(cudaSetDevice(device));
#elif defined(__HIP_PLATFORM_HCC__)
      cudaCheck(hipSetDevice(device));
#else
#endif
    }
  }

  int original_device = -1;

private:
#ifdef WITH_CUDA
  static void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
      std::string msg = "CUDA error (";
      msg += std::to_string(err);
      msg += "): ";
      msg += cudaGetErrorString(err);
      throw std::runtime_error(msg);
    }
  }
#elif defined(__HIP_PLATFORM_HCC__)
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
#endif
};
