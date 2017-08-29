#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct LeakyReLUUpdateOutput
{
  const T negval_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
  LeakyReLUUpdateOutput() = default;

  __host__ __device__
  LeakyReLUUpdateOutput(const LeakyReLUUpdateOutput& o) = default;

  __host__ __device__
  ~LeakyReLUUpdateOutput() {}

  __host__ __device__
#endif
  LeakyReLUUpdateOutput(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : x * negval_;
  }
};

// in-place variant
template <typename T>
struct LeakyReLUUpdateOutputIP
{
  const T negval_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
  LeakyReLUUpdateOutputIP() = default;

  __host__ __device__
  LeakyReLUUpdateOutputIP(const LeakyReLUUpdateOutputIP& r) = default;

  __host__ __device__
  ~LeakyReLUUpdateOutputIP() {}

  __host__ __device__
#endif
  LeakyReLUUpdateOutputIP(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    *x = (*x > 0) ? *x : negval_ * (*x);
  }
};

template <typename T>
struct LeakyReLUUpdateGradInput
{
  const T negval_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
  LeakyReLUUpdateGradInput() = default;

  __host__ __device__
  LeakyReLUUpdateGradInput(const LeakyReLUUpdateGradInput& f) = default;

  __host__ __device__
  ~LeakyReLUUpdateGradInput() {}

  __host__ __device__
#endif
  LeakyReLUUpdateGradInput(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    T* gradInput,
    T* input,
    T* gradOutput) const
  {
    *gradInput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }
};

template <typename T>
struct LeakyReLUUpdateGradInputIP
{
  const T negval_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
  LeakyReLUUpdateGradInputIP() = default;

  __host__ __device__
  LeakyReLUUpdateGradInputIP(const LeakyReLUUpdateGradInputIP& t) = default;

  __host__ __device__
  ~LeakyReLUUpdateGradInputIP() {}

  __host__ __device__
#endif
  LeakyReLUUpdateGradInputIP(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    T* gradOutput,
    T* input) const
  {
    *gradOutput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }
};

#include "generic/LeakyReLU.cu"
#include "THCGenerateFloatTypes.h"
