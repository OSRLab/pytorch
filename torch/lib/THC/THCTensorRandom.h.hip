#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

#include "generic/THCTensorRandom.h"
#include "THCGenerateAllTypes.h"

#include <hiprand_kernel.h>

#define MTGP32_STATE_SIZE 1024    // The effective state number is =351

// Structure of hcrngStateMtgp32
typedef struct hcrngStateMtgp32 {
  uint32_t s[MTGP32_STATE_SIZE];
  uint32_t offset;
  uint32_t pIdx;
  mtgp32_kernel_params_t* k;  // points to all kernel parameters
} hcrngStateMtgp32;

/* Generator */
struct Generator {
  //hcrngStateMtgp32* gen_states;
  hiprandStateMtgp32_t* gen_states;
  mtgp32_kernel_params_t *kernel_params;
  int initf;
  uint64_t initial_seed;
};

typedef struct THCRNGState {
  /* One generator per GPU */
  Generator* gen;
  int num_devices;
} THCRNGState;

struct THCState;

THC_API void THCRandom_init(THCState *state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(THCState *state);
THC_API uint64_t THCRandom_seed(THCState *state);
THC_API uint64_t THCRandom_seedAll(THCState *state);
THC_API void THCRandom_manualSeed(THCState *state, uint64_t the_seed_);
THC_API void THCRandom_manualSeedAll(THCState *state, uint64_t the_seed_);
THC_API uint64_t THCRandom_initialSeed(THCState *state);
THC_API void THCRandom_getRNGState(THCState *state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(THCState *state, THByteTensor *rng_state);

THC_API hiprandStateMtgp32_t* THCRandom_generatorStates(THCState* state);

#endif
