#ifndef THC_TENSORMATH_CUH
#define THC_TENSORMATH_CUH

// Copy the kth diagonal of a matrix B to a vector A.
template <typename T>
__global__ void THCTensor_copyFromDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
template <typename T>
__global__ void THCTensor_copyToDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

#define CAT_ARRAY_BATCH_SIZE 1024
#define CAT_ARRAY_MAX_INPUT_DIMS 4

inline bool getCatGrid(THCState* state, ptrdiff_t nTensors, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);

  if (curDevice == -1) {
     return false;
  }

  // Assume a reasonable number of SMs if no state is available
  int numSM =
        state ? THCState_getCurrentDeviceProperties(state)->multiProcessorCount : 15;
  //X dim of grid for cat array cooperates on a single tensor in the cat.
  //Given half of the GPU, full utilization will always occur.
  grid = dim3( 2LL * numSM, (long long) nTensors );
	     
  return true;
}

// Similar to any other IndexToOffset calculation for copying along a given dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : outputSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * outputStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * outputStride[0];
  }
};

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template<typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
#if defined(__HIP_PLATFORM_HCC__)
#define MAX_Dim 6
  IndexType outputSize[MAX_Dim];
  IndexType outputStride[MAX_Dim];

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    assert(MAX_Dim == 6); // This is hardcoded into the deserialize function signature and the mapping below
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(outputSize[0]), &outputSize[i]);
    }
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(outputStride[0]), &outputStride[i]);
    }
  }

  OutputTensorSizeStride() {
    for (int i=0; i<MAX_Dim; i++) {
      outputSize[i] = 0;
      outputStride[i] = 0;
    }
  }

  __attribute__((annotate("user_deserialize")))
  OutputTensorSizeStride(
                 IndexType size0, IndexType size1, IndexType size2, IndexType size3, IndexType size4, IndexType size5,
                 IndexType stride0, IndexType stride1, IndexType stride2, IndexType stride3, IndexType stride4, IndexType stride5
                 ) [[cpu]][[hc]] {

    outputSize[0]   = size0;
    outputSize[1]   = size1;
    outputSize[2]   = size2;
    outputSize[3]   = size3;
    outputSize[4]   = size4;
    outputSize[5]   = size5;
    outputStride[0] = stride0;
    outputStride[1] = stride1;
    outputStride[2] = stride2;
    outputStride[3] = stride3;
    outputStride[4] = stride4;
    outputStride[5] = stride5;
  }
#else
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
#endif
};

/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a grid-stride loop based off of
  * the blockIdx.x, threadIdx.x for each input to copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */



template <typename T, typename IndexType, int Dims>
__global__ void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs[blockIdx.y].nElements;

    if(tid >= nElements) return;
    
    T* data = inputs[blockIdx.y].input;
    IndexType offset = inputs[blockIdx.y].offset;
    IndexType dimSize = inputs[blockIdx.y].dimSize;
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
    IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
    	      os.outputSize, os.outputStride, dimSize, concatDim, tid);
    output[dataOffset + elementOffset] = data[tid];

    tid += stride;
    }
}

#endif
