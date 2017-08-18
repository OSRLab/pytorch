#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

template <typename T, typename TOut>
struct TensorLTValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorLTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorLTValueOp() {}
#endif

  const T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorGTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorGTValueOp() {}
#endif

  const T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorLEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorLEValueOp() {}
#endif

  const T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorGEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorGEValueOp() {}
#endif

  const T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorEQValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorEQValueOp() {}
#endif

  const T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__
#endif
  TensorNEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*in, value));
  }

#if defined(__HIP_PLATFORM_HCC__)
  __host__ __device__ ~TensorNEValueOp() {}
#endif

  const T value;
};

template<typename TensorType, typename TensorTypeOut, class Op>
void THC_logicalValue(THCState *state,
                      TensorTypeOut *self_,
                      TensorType *src,
                      Op op) {
  THLongStorage* st = TensorUtils<TensorType>::newSizeOf(state, src);
  TensorUtils<TensorTypeOut>::resize(state, self_, st, NULL);
  THLongStorage_free(st);

  if (!THC_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_TENSORMATH_COMPARE_CUH
