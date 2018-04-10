<<<<<<< HEAD
=======
#include "hip/hip_runtime.h"
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorRandom.cu"
#else

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
<<<<<<< HEAD

=======
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
THC_API void THCTensor_(uniform)(THCState* state, THCTensor *self_, double a, double b)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_uniform<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, a, b);
=======
  hipLaunchKernelGGL((generate_uniform), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, a, b);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(normal)(THCState* state, THCTensor *self_, double mean, double stdv)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, mean, stdv);
=======
  hipLaunchKernelGGL((generate_normal), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, mean, stdv);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(normal_means)(THCState *state, THCTensor *self, THCTensor *means, double stddev) {
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, stddev);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, real>::to(1), means);
}

THC_API void THCTensor_(normal_stddevs)(THCState *state, THCTensor *self, double mean, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, stddevs);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(add)(state, self, self, ScalarConvert<double, real>::to(mean));
}

THC_API void THCTensor_(normal_means_stddevs)(THCState *state, THCTensor *self, THCTensor *means, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, real>::to(1), means);
}

THC_API void THCTensor_(logNormal)(THCState* state, THCTensor *self_, double mean, double stdv)
{

  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generateLogNormal<real><<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, mean, stdv);
=======
  hipLaunchKernelGGL((generateLogNormal<real>), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, mean, stdv);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(exponential)(THCState* state, THCTensor *self_, double lambda)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_exponential<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, lambda);
=======
  hipLaunchKernelGGL((generate_exponential), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, lambda);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(cauchy)(THCState* state, THCTensor *self_, double median, double sigma)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_cauchy<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, median, sigma);
=======
  hipLaunchKernelGGL((generate_cauchy), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, median, sigma);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(renormRows)(struct THCState* state,
                             THCTensor* t) {
  THAssert(THCTensor_(nDimension)(state, t) == 2);
<<<<<<< HEAD
  int64_t rows = THCTensor_(size)(state, t, 0);
  int64_t cols = THCTensor_(size)(state, t, 1);

  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
=======
  long rows = THCTensor_(size)(state, t, 0);
  long cols = THCTensor_(size)(state, t, 1);

  hipDeviceProp_t* props = THCState_getCurrentDeviceProperties(state);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

<<<<<<< HEAD
  renormRowsL1<real>
    <<<grid, block, block.x * sizeof(real),
    THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, t),
=======
  hipLaunchKernelGGL((renormRowsL1<real>), dim3(grid), dim3(block), block.x * sizeof(real), THCState_getCurrentStream(state), THCTensor_(data)(state, t),
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
                                        rows, cols);
}

THC_API void THCTensor_(multinomial)(struct THCState *state,
                                      THCudaLongTensor *self,
                                      THCTensor *prob_dist,
                                      int n_sample,
                                      int with_replacement)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, prob_dist));
  THCGenerator* gen = THCRandom_getGenerator(state);

  int inputSize = THCTensor_(nDimension)(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
<<<<<<< HEAD
  int64_t numDist =
    inputSize == 1 ? 1 : THCTensor_(size)(state, prob_dist, 0);
  int64_t numCategoriesLong =
=======
  long numDist =
    inputSize == 1 ? 1 : THCTensor_(size)(state, prob_dist, 0);
  long numCategoriesLong =
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
    inputSize == 1 ? THCTensor_(size)(state, prob_dist, 0) :
    THCTensor_(size)(state, prob_dist, 1);

  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  THArgCheck(numCategoriesLong <= FLOAT32_MAX_CONSECUTIVE_INT, 2,
             "number of categories cannot exceed 2^24");
  int numCategories = (int) numCategoriesLong;

  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");

  if (!with_replacement) {
    THArgCheck(n_sample <= numCategories, 2,
               "cannot sample n_sample > prob_dist:size(1) samples without "
               "replacement");
  }

<<<<<<< HEAD
  int free_prob_dist = 0;

  // Restructure data for 2d
  if (inputSize == 1) {
    THCTensor *temp = THCTensor_(new)(state);
    THCTensor_(unsqueeze1d)(state, temp, prob_dist, 0);
    prob_dist = temp;
    free_prob_dist = 1;
=======
  // It is possible that prob_dist is non-contiguous
  THCTensor* probDistContig =
    THCTensor_(newContiguous)(state, prob_dist);

  // Restructure data for 2d
  if (inputSize == 1) {
    THCTensor_(resize2d)(state, probDistContig, 1, numCategories);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
  }

  THCudaLongTensor_resize2d(state, self, numDist, n_sample);

  // get current device properties
<<<<<<< HEAD
  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
=======
  hipDeviceProp_t* props = THCState_getCurrentDeviceProperties(state);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
  THAssert(props != NULL);
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;
  int maxShared = props->sharedMemPerBlock;
  int requiredShared = (numCategories < maxThreads ? numCategories : maxThreads)
                                * (sizeof(real) * sizeof(accreal));

  if (n_sample == 1 && maxShared >= requiredShared) {
    // Optimized allocation-free implementation
    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into
    // temporarily allocated memory. The device RNG is thread-limited
    THCTensor *sampled = THCTensor_(newWithSize2d)(state, numDist, n_sample);
    THCTensor_(uniform)(state, sampled, 0.0, 1.0);

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

<<<<<<< HEAD
    sampleMultinomialOnce<real, accreal>
      <<<grid, block,
         requiredShared,
         THCState_getCurrentStream(state)>>>(
=======
    hipLaunchKernelGGL((sampleMultinomialOnce<real, accreal>), dim3(grid), dim3(block), requiredShared, THCState_getCurrentStream(state), 
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
      THCudaLongTensor_data(state, self),
      numDist,
      numCategories,
      THCTensor_(data)(state, sampled),
<<<<<<< HEAD
      THCTensor_(data)(state, prob_dist),
      THCTensor_(stride)(state, prob_dist, 0),
      THCTensor_(stride)(state, prob_dist, 1)
      );
=======
      THCTensor_(data)(state, probDistContig));
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
    THCTensor_(free)(state, sampled);
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCTensor* origDist = THCTensor_(new)(state);
<<<<<<< HEAD
    THCTensor_(resizeAs)(state, origDist, prob_dist);
    THCTensor_(copy)(state, origDist, prob_dist);

    THCTensor* normDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, normDist, prob_dist);
=======
    THCTensor_(resizeAs)(state, origDist, probDistContig);
    THCTensor_(copy)(state, origDist, probDistContig);

    THCTensor* normDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, normDist, probDistContig);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

    THCTensor* prefixSum = THCTensor_(new)(state);

    // Renorm along rows
    THCTensor_(copy)(state, normDist, origDist);
    THCTensor_(renormRows)(state, normDist);

    // Prefix sum along rows
    THCTensor_(cumsum)(state, prefixSum, normDist, 1);

    if (with_replacement) {
      // Sample with replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from one
      // distribution concurrently.
      dim3 grid(numDist < MAX_NUM_BLOCKS ? numDist : MAX_NUM_BLOCKS);

<<<<<<< HEAD
      sampleMultinomialWithReplacement
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
=======
      hipLaunchKernelGGL((sampleMultinomialWithReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
          gen->gen_states,
          n_sample,
          THCudaLongTensor_data(state, self),
          numDist, numCategories,
          THCTensor_(data)(state, prefixSum));
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
<<<<<<< HEAD
      ptrdiff_t numBlocks = THCCeilDiv(numDist, (int64_t) 4);
=======
      ptrdiff_t numBlocks = THCCeilDiv(numDist, 4L);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
      dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

      for (int sample = 0; sample < n_sample; ++sample) {
        if (sample > 0) {
          // Update probabilities
          // Renorm along rows
          THCTensor_(copy)(state, normDist, origDist);
          THCTensor_(renormRows)(state, normDist);

          // Prefix sum along rows
          THCTensor_(cumsum)(state, prefixSum, normDist, 1);
        }

        // The kernel can only draw one sample before we have to
        // recalculate our distribution
<<<<<<< HEAD
        sampleMultinomialWithoutReplacement
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
=======
        hipLaunchKernelGGL((sampleMultinomialWithoutReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
            gen->gen_states,
            n_sample,
            sample,
            THCudaLongTensor_data(state, self),
            numDist, numCategories,
            THCTensor_(data)(state, origDist),
            THCTensor_(data)(state, prefixSum));
      }
    }

    THCTensor_(free)(state, prefixSum);
    THCTensor_(free)(state, normDist);
    THCTensor_(free)(state, origDist);
  }

  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCudaLongTensor_resize1d(state, self, n_sample);
<<<<<<< HEAD
  }
  if (free_prob_dist) {
    THCTensor_(free)(state, prob_dist);
  }
=======

    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCTensor_(resize1d)(state, probDistContig, numCategories);
  }

  THCTensor_(free)(state, probDistContig);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
}

THC_API void THCTensor_(multinomialAliasSetup)(THCState *state, THCTensor *_probs, THCudaLongTensor *_J, THCTensor *_q){
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THAssert(THCTensor_(isContiguous)(state, _probs));
<<<<<<< HEAD
  int64_t inputsize = THCTensor_(nElement)(state, _probs);
=======
  long inputsize = THCTensor_(nElement)(state, _probs);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
  THCudaLongTensor *smaller = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *smaller_short = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger_short = THCudaLongTensor_newWithSize1d(state, inputsize);

  THCudaLongTensor_resize1d(state, _J, inputsize);
  THCTensor_(resize1d)(state, _q, inputsize);

<<<<<<< HEAD
  real one = ScalarConvert<int64_t, real>::to(1);
  int inputBlockDim = THCCeilDiv((int)inputsize + BLOCK_SIZE - 1, BLOCK_SIZE);
  aliasMultinomialFilter
    <<<inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state) >>>(
                     THCTensor_(data)(state, _q),
                     THCTensor_(data)(state, _probs),
                     THCudaLongTensor_data(state, smaller),
                     THCudaLongTensor_data(state, larger),
                     THCudaLongTensor_data(state, _J),
                     THCudaLongTensor_data(state, smaller_short),
                     THCudaLongTensor_data(state, larger_short),
                     one, inputsize
                     );
=======
  real one = ScalarConvert<long, real>::to(1);
  int inputBlockDim = THCCeilDiv((int)inputsize + BLOCK_SIZE - 1, BLOCK_SIZE);
  hipLaunchKernelGGL((aliasMultinomialFilter), dim3(inputBlockDim), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state) , 
								     THCTensor_(data)(state, _q),
								     THCTensor_(data)(state, _probs),
								     THCudaLongTensor_data(state, smaller),
								     THCudaLongTensor_data(state, larger),
								     THCudaLongTensor_data(state, _J),
								     THCudaLongTensor_data(state, smaller_short),
								     THCudaLongTensor_data(state, larger_short),
								     one, inputsize
								     );
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCudaLongTensor_nonzero(state, smaller_short, smaller);
  THCudaLongTensor_nonzero(state, larger_short, larger);
  int h_large_c = THCudaLongTensor_nElement(state, larger_short);
  THCudaLongTensor_resize1d(state, smaller_short, inputsize);
  THCudaLongTensor_resize1d(state, larger_short, inputsize);
<<<<<<< HEAD
  aliasMultinomialSetup
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
                THCudaLongTensor_data(state, _J),
                THCTensor_(data)(state, _q),
                inputsize,
                THCudaLongTensor_data(state, smaller_short),
                THCudaLongTensor_data(state, larger_short),
                inputsize - h_large_c, h_large_c
                );
  real q_max = THCTensor_(maxall)(state, _q);
  condDiv<<<
    inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
                      THCTensor_(data)(state, _q),
                      THCudaLongTensor_data(state, _J),
                      inputsize, q_max
                      );
=======
  hipLaunchKernelGGL((aliasMultinomialSetup), dim3(1), dim3(1), 0, THCState_getCurrentStream(state), 
						    THCudaLongTensor_data(state, _J),
						    THCTensor_(data)(state, _q),
						    inputsize,
						    THCudaLongTensor_data(state, smaller_short),
						    THCudaLongTensor_data(state, larger_short),
						    static_cast<int>(inputsize - h_large_c), h_large_c
						    );
  real q_max = THCTensor_(maxall)(state, _q);
  hipLaunchKernelGGL((condDiv), dim3(inputBlockDim), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
								      THCTensor_(data)(state, _q),
								      THCudaLongTensor_data(state, _J),
								      inputsize, q_max
								      );
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCudaLongTensor_free(state, smaller);
  THCudaLongTensor_free(state, larger);
  THCudaLongTensor_free(state, smaller_short);
  THCudaLongTensor_free(state, larger_short);
}

THC_API void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCudaLongTensor *_J, THCTensor *_q){
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THCGenerator* gen = THCRandom_getGenerator(state);
<<<<<<< HEAD
  int64_t K = THCudaLongTensor_nElement(state, _J);
  int64_t output_nelem = THCudaLongTensor_nElement(state, self);
=======
  long K = THCudaLongTensor_nElement(state, _J);
  long output_nelem = THCudaLongTensor_nElement(state, self);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
  ptrdiff_t size = THCudaLongTensor_nElement(state, self);

  THCTensor *uniform = THCTensor_(newWithSize1d)(state, output_nelem);
  THCTensor *bernoulli = THCTensor_(newWithSize1d)(state, output_nelem);

  THCTensor_(uniform)(state, uniform, 0, K);
  THCTensor_(uniform)(state, bernoulli, 0, 1);

<<<<<<< HEAD
  multinomialAliasDrawKernel
    <<<THCCeilDiv((int)output_nelem+BLOCK_SIZE-1, BLOCK_SIZE), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
          size,
          THCudaLongTensor_data(state, self),
          THCudaLongTensor_data(state, _J),
          THCTensor_(data)(state, _q),
          K,
          THCTensor_(data)(state, uniform),
          THCTensor_(data)(state, bernoulli)
          );
=======
  hipLaunchKernelGGL((multinomialAliasDrawKernel), dim3(THCCeilDiv((int)output_nelem+BLOCK_SIZE-1, BLOCK_SIZE)), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
				  static_cast<int>(size),
				  THCudaLongTensor_data(state, self),
				  THCudaLongTensor_data(state, _J),
				  THCTensor_(data)(state, _q),
				  K,
				  THCTensor_(data)(state, uniform),
				  THCTensor_(data)(state, bernoulli)
				  );
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
}

THC_API void THCTensor_(rand)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(uniform)(state, r_, 0, 1);
}

void THCTensor_(randn)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(normal)(state, r_, 0, 1);
}

#endif

#if defined(THC_REAL_IS_DOUBLE)
<<<<<<< HEAD
GENERATE_KERNEL1(generate_bernoulli, double, double p, double, curand_uniform_double, x <= p)
#else
GENERATE_KERNEL1(generate_bernoulli, real, double p, float, curand_uniform, (ScalarConvert<bool, real>::to(x <= p)))
=======
GENERATE_KERNEL1(generate_bernoulli, double, double p, double, hiprng_uniform_double, x <= p)
#else
GENERATE_KERNEL1(generate_bernoulli, real, double p, float, hiprng_uniform, (ScalarConvert<bool, real>::to(x <= p)))
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#endif

THC_API void THCTensor_(bernoulli)(THCState* state, THCTensor *self_, double p)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_bernoulli<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);
=======
  hipLaunchKernelGGL((generate_bernoulli), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, p);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(bernoulli_Tensor)(THCState *state, THCTensor *self, THCTensor* p)
{
<<<<<<< HEAD
#if defined(THC_REAL_IS_FLOAT)
  THCTensor_(bernoulli_FloatTensor)(state, self, p);
#elif defined(THC_REAL_IS_DOUBLE)
  THCTensor_(bernoulli_DoubleTensor)(state, self, p);
#endif
}

#define DEFINE_BERNOULLI_TENSOR(NAME, PROB_TYPE, PROB_DATA_TYPE)               \
THC_API void THCTensor_(NAME)(THCState* state,                                 \
        THCTensor *self_, PROB_TYPE *probs_)                                   \
{                                                                              \
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, probs_));             \
  ptrdiff_t size = THCTensor_(nElement)(state, self_);                         \
  if (size == 0) return;                                                       \
  THCGenerator* gen = THCRandom_getGenerator(state);                           \
  THCTensor *self = THCTensor_(newContiguous)(state, self_);                   \
  PROB_TYPE *probs = PROB_TYPE##_newContiguous(state, probs_);                 \
  ptrdiff_t prob_size = PROB_TYPE##_nElement(state, probs);                    \
  real *result_data = THCTensor_(data)(state, self);                           \
  PROB_DATA_TYPE *probs_data = PROB_TYPE##_data(state, probs);                 \
                                                                               \
  THArgCheck(size == prob_size, 3, "inconsistent tensor size");                \
                                                                               \
  generate_bernoulli_tensor<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>( \
      gen->gen_states, size, result_data, probs_data);                         \
                                                                               \
  PROB_TYPE##_free(state, probs);                                              \
  THCTensor_(freeCopyTo)(state, self, self_);                                  \
}
=======
 #if defined(THC_REAL_IS_FLOAT)
   THCTensor_(bernoulli_FloatTensor)(state, self, p);
 #elif defined(THC_REAL_IS_DOUBLE)
   THCTensor_(bernoulli_DoubleTensor)(state, self, p);
 #endif
}

#if defined(__HIP_PLATFORM_HCC__)
  #define DEFINE_BERNOULLI_TENSOR(NAME, PROB_TYPE, PROB_DATA_TYPE)               \
  THC_API void THCTensor_(NAME)(THCState* state,                                 \
          THCTensor *self_, PROB_TYPE *probs_)                                   \
  {                                                                              \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, probs_));             \
    ptrdiff_t size = THCTensor_(nElement)(state, self_);                         \
    if (size == 0) return;                                                       \
    THCGenerator* gen = THCRandom_getGenerator(state);                           \
    THCTensor *self = THCTensor_(newContiguous)(state, self_);                   \
    PROB_TYPE *probs = PROB_TYPE##_newContiguous(state, probs_);                 \
    ptrdiff_t prob_size = PROB_TYPE##_nElement(state, probs);                    \
    real *result_data = THCTensor_(data)(state, self);                           \
    PROB_DATA_TYPE *probs_data = PROB_TYPE##_data(state, probs);                 \
                                                                                 \
    THArgCheck(size == prob_size, 3, "inconsistent tensor size");                \
                                                                                 \
    hipLaunchKernelGGL(                                                          \
      (generate_bernoulli_tensor), NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state), \
        gen->gen_states, static_cast<int>(size), result_data, probs_data);       \
                                                                                 \
    PROB_TYPE##_free(state, probs);                                              \
    THCTensor_(freeCopyTo)(state, self, self_);                                  \
  }
#else
  #define DEFINE_BERNOULLI_TENSOR(NAME, PROB_TYPE, PROB_DATA_TYPE)               \
  THC_API void THCTensor_(NAME)(THCState* state,                                 \
          THCTensor *self_, PROB_TYPE *probs_)                                   \
  {                                                                              \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, probs_));             \
    ptrdiff_t size = THCTensor_(nElement)(state, self_);                         \
    if (size == 0) return;                                                       \
    THCGenerator* gen = THCRandom_getGenerator(state);                              \
    THCTensor *self = THCTensor_(newContiguous)(state, self_);                   \
    PROB_TYPE *probs = PROB_TYPE##_newContiguous(state, probs_);                 \
    ptrdiff_t prob_size = PROB_TYPE##_nElement(state, probs);                    \
    real *result_data = THCTensor_(data)(state, self);                           \
    PROB_DATA_TYPE *probs_data = PROB_TYPE##_data(state, probs);                 \
                                                                                 \
    THArgCheck(size == prob_size, 3, "inconsistent tensor size");                \
                                                                                 \
    hipLaunchKernelGGL((generate_bernoulli_tensor), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state),  \
        gen->gen_states, size, result_data, probs_data);                         \
                                                                                 \
    PROB_TYPE##_free(state, probs);                                              \
    THCTensor_(freeCopyTo)(state, self, self_);                                  \
  }
#endif
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

DEFINE_BERNOULLI_TENSOR(bernoulli_FloatTensor, THCudaTensor, float)
DEFINE_BERNOULLI_TENSOR(bernoulli_DoubleTensor, THCudaDoubleTensor, double)

#if defined(THC_REAL_IS_DOUBLE)
<<<<<<< HEAD
GENERATE_KERNEL1(generate_geometric, double, double p, double, curand_uniform_double, ceil(log(x) / log(1-p)))
#else
GENERATE_KERNEL1(generate_geometric, real, double p, float, curand_uniform, (ScalarConvert<float, real>::to(ceilf(logf(x) / log(1-p)))))
#endif

#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
#define CURAND64(STATE) (((uint64_t)curand(&state[blockIdx.x])) << 32) | (uint64_t)curand(&state[blockIdx.x])
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, curand, (real)(x % range + base))
GENERATE_KERNEL2(generate_random_64, real, int64_t base, uint64_t range, uint64_t, CURAND64, (real)(x % range + base))
#elif defined(THC_REAL_IS_HALF)
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, curand, (ScalarConvert<uint32_t, real>::to(x % range + base)))
#else
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, curand, (real)(x % range + base))
=======
GENERATE_KERNEL1(generate_geometric, double, double p, double, hiprng_uniform_double, ceil(log(x) / log(1-p)))
#else
GENERATE_KERNEL1(generate_geometric, real, double p, float, hiprng_uniform, (ScalarConvert<float, real>::to(ceilf(logf(x) / log(1-p)))))
#endif

#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
#define CURAND64(STATE) (((uint64_t) hiprng(&state[blockIdx.x])) << 32) | (uint64_t) hiprng(&state[blockIdx.x])
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, hiprng, (real)(x % range + base))
GENERATE_KERNEL2(generate_random_64, real, int64_t base, uint64_t range, uint64_t, CURAND64, (real)(x % range + base))
#elif defined(THC_REAL_IS_HALF)
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, hiprng, (ScalarConvert<uint32_t, real>::to(x % range + base)))
#else
GENERATE_KERNEL2(generate_random, real, int32_t base, uint32_t range, uint32_t, hiprng, (real)(x % range + base))
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#endif

THC_API void THCTensor_(geometric)(THCState* state, THCTensor *self_, double p)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

<<<<<<< HEAD
  generate_geometric<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);
=======
  hipLaunchKernelGGL((generate_geometric), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, p);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(clampedRandom)(THCState* state, THCTensor *self_, int64_t min_val, int64_t max_val)
{
  THArgCheck(min_val < max_val, 2,
             "max must be greater than min, but got: min = %lld, max = %lld", min_val, max_val);
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

  uint64_t range = max_val - min_val;

#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
  if (range > 1ULL << 32) {
<<<<<<< HEAD
    generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        gen->gen_states, size, data, min_val, range);
  } else {
#endif
    generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        gen->gen_states, size, data, min_val, range);
=======
    hipLaunchKernelGGL((generate_random_64), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
        gen->gen_states, static_cast<int>(size), data, static_cast<int64_t>(min_val), static_cast<uint64_t>(range));
  } else {
#endif
    hipLaunchKernelGGL((generate_random), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
        gen->gen_states, static_cast<int>(size), data, static_cast<int32_t>(min_val), static_cast<uint32_t>(range));
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
  }
#endif

  THCTensor_(freeCopyTo)(state, self, self_);
};

THC_API void THCTensor_(cappedRandom)(THCState* state, THCTensor *self_, int64_t max_val)
{
  THCTensor_(clampedRandom)(state, self_, 0LL, max_val);
};

#define HLF_MANT_DIG 11

THC_API void THCTensor_(random)(THCState* state, THCTensor *self_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  real *data = THCTensor_(data)(state, self);

#if defined(THC_REAL_IS_HALF)
<<<<<<< HEAD
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, 0UL, (1UL << HLF_MANT_DIG) + 1);
#elif defined(THC_REAL_IS_FLOAT)
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, 0UL, (1UL << FLT_MANT_DIG) + 1);
#elif defined(THC_REAL_IS_DOUBLE)
  generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, 0ULL, (1ULL << DBL_MANT_DIG) + 1);
#elif defined(THC_REAL_IS_LONG)
  generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, 0ULL, static_cast<uint64_t>(std::numeric_limits<real>::max()) + 1);
#else
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, 0UL, static_cast<uint32_t>(std::numeric_limits<real>::max()) + 1);
=======
  hipLaunchKernelGGL((generate_random), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>((1UL << HLF_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_FLOAT)
  hipLaunchKernelGGL((generate_random), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>((1UL << FLT_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_DOUBLE)
  hipLaunchKernelGGL((generate_random_64), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, static_cast<int64_t>(0ULL), static_cast<uint64_t>((1ULL << DBL_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_LONG)
  hipLaunchKernelGGL((generate_random_64), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, static_cast<int64_t>(0ULL), static_cast<uint64_t>(std::numeric_limits<real>::max()) + 1);
#else
  hipLaunchKernelGGL((generate_random), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>(std::numeric_limits<real>::max()) + 1);
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#endif

  THCTensor_(freeCopyTo)(state, self, self_);
};
<<<<<<< HEAD

#undef HLF_MANT_DIG
#undef CURAND64
=======
>>>>>>> c4e8c2866e4224c83592336cd609139311fc3854
#undef NUM_BLOCKS

#endif
