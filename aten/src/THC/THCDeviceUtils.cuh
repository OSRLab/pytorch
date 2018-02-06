#ifndef THC_DEVICE_UTILS_INC
#define THC_DEVICE_UTILS_INC

/* The largest consecutive integer representable in float32 (2^24) */
#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
__host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
  return THCCeilDiv(a, b) * b;
}

/**
 * For CC 3.5+, perform a load using __ldg
 */
template <typename T>
#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline T doLdg(const T* p) {
#else
__device__ __forceinline__ T doLdg(const T* p) {
#endif
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline unsigned int ACTIVE_MASK()
#else
__device__ __forceinline__ unsigned int ACTIVE_MASK()
#endif
{
#if CUDA_VERSION >= 9000
    return __activemask();
#else
// will be ignored anyway
    return 0xffffffff;
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
#else
__device__ __forceinline__ int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
#endif
{
#if CUDA_VERSION >= 9000
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline double WARP_SHFL_XOR(double value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return (double)__shfl_xor((float)value, laneMask, width);
}
#endif

template <typename T>
#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
#else
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
#endif
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
#else
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
#endif
{
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T>
#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
#else
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
#endif
{
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(mask, value, delta, width);
#else
    return __shfl_up(value, delta, width);
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ double WARP_SHFL_DOWN(double value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return (double)__shfl_down((float)value, delta, width);
}
#endif

template <typename T>
#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ inline T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
#else
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
#endif
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}


#endif // THC_DEVICE_UTILS_INC
