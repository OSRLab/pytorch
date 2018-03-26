#pragma once

// RAII structs to set CUDA stream

#if defined(WITH_CUDA) || defined(WITH_ROCM)
#include <THC/THC.h>
extern THCState* state;
#endif

struct AutoStream {
#if defined(WITH_CUDA) || defined(WITH_ROCM)
  explicit AutoStream(THCStream* stream)
    : original_stream(THCState_getStream(state))
  {
    THCStream_retain(original_stream);
    THCState_setStream(state, stream);
  }

  ~AutoStream() {
    THCState_setStream(state, original_stream);
    THCStream_free(original_stream);
  }

  THCStream* original_stream;
#endif
};
