#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  const T alpha_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ 
  ELUupdateOutput_functor() = default;

  __host__ __device__ 
  ELUupdateOutput_functor(const ELUupdateOutput_functor& f) = default;

  __host__ __device__ 
  ~ELUupdateOutput_functor() {}

  __host__ __device__ 
#endif
  ELUupdateOutput_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ : *input;
  }
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T alpha_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ 
  ELUupdateOutputIP_functor() = default;

  __host__ __device__ 
  ELUupdateOutputIP_functor(const ELUupdateOutputIP_functor& f) = default;

  __host__ __device__ 
  ~ELUupdateOutputIP_functor() {}

  __host__ __device__ 
#endif
  ELUupdateOutputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * alpha_ : *x;
  }
};

template <typename T>
struct ELUupdateGradInput_functor
{
  const T alpha_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ 
  ELUupdateGradInput_functor() = default;

  __host__ __device__ 
  ELUupdateGradInput_functor(const ELUupdateGradInput_functor& f) = default;

  __host__ __device__ 
  ~ELUupdateGradInput_functor() {}

  __host__ __device__ 
#endif
  ELUupdateGradInput_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

template <typename T>
struct ELUupdateGradInputIP_functor
{
  const T alpha_;

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ 
  ELUupdateGradInputIP_functor() = default;

  __host__ __device__ 
  ELUupdateGradInputIP_functor(const ELUupdateGradInputIP_functor& f) = default;

  __host__ __device__ 
  ~ELUupdateGradInputIP_functor() {}

  __host__ __device__ 
#endif
  ELUupdateGradInputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *gradOutput, const T *output) const
  {
    *gradOutput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

#include "generic/ELU.cu"
#include "THCGenerateFloatTypes.h"
