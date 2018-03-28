#pragma once

// RAII structs to set CUDA device

#include <string>
#include <stdexcept>

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
    if (original_device != -1) {
      cudaSetDevice(original_device);
    }
  }

  inline void setDevice(int device) {
    if (device == -1) {
      return;
    }

    if (original_device == -1) {
      cudaCheck(cudaGetDevice(&original_device));
      if (device != original_device) {
        cudaCheck(cudaSetDevice(device));
      }
    } else {
      cudaCheck(cudaSetDevice(device));
    }
  }

  int original_device = -1;

private:
  static void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
      std::string msg = "CUDA error (";
      msg += std::to_string(err);
      msg += "): ";
      msg += cudaGetErrorString(err);
      throw std::runtime_error(msg);
    }
  }
};
