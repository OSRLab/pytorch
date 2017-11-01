#include "THCTensorSort.cuh"

void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim) {
  int64_t dims = THCudaLongTensor_nDimension(state, t);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  ptrdiff_t inElements = THCudaLongTensor_nElement(state, t);
  int64_t sliceSize = THCudaLongTensor_size(state, t, dim);
  ptrdiff_t numSlices = inElements / sliceSize;

  dim3 grid;
  if (!THC_getGridFromTiles(numSlices, grid)) {
    THError("Slice to fill with indices is too large");
  }

  int64_t maxThreads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  int64_t numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);

#if defined(__HIP_PLATFORM_HCC__)
  #define FILL_INDEX(T, DIM)                                       \
    hipLaunchKernelGGL(                                            \
      (fillSliceWithIndex<T, DIM>),                                \
        grid, block, 0, THCState_getCurrentStream(state),          \
          make_magic_wrapper(info), numSlices, sliceSize,          \
          info.strides[collapseDim])
#else
  #define FILL_INDEX(T, DIM)                                       \
    fillSliceWithIndex<T, DIM>                                     \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
        info, numSlices, sliceSize, info.strides[collapseDim])
#endif

  if (TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, t)) {
    TensorInfo<int64_t, uint32_t> info =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      FILL_INDEX(unsigned int, -2);
    } else {
      if (info.dims == 1) {
        FILL_INDEX(unsigned int, 1);
      } else if (info.dims == 2) {
        FILL_INDEX(unsigned int, 2);
      } else {
        FILL_INDEX(unsigned int, -1);
      }
    }
  } else {
    TensorInfo<int64_t, uint64_t> info =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    FILL_INDEX(uint64_t, -1);
  }

#undef FILL_INDEX

  THCudaCheck(cudaGetLastError());
}
