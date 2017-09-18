#ifndef THC_STREAM_INC
#define THC_STREAM_INC

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h>
#endif
#include "THCGeneral.h"

struct THCStream
{
    cudaStream_t stream;
    int device;
    int refcount;
};


THC_API THCStream* THCStream_new(int flags);
THC_API THCStream* THCStream_defaultStream(int device);
THC_API THCStream* THCStream_newWithPriority(int flags, int priority);
THC_API void THCStream_free(THCStream* self);
THC_API void THCStream_retain(THCStream* self);

#endif // THC_STREAM_INC
