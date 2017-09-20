#include "THC.h"
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCHalf.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCDeviceUtils.cuh"
#include "THCNumerics.cuh"
#include "THCAtomics.cuh"
#include <algorithm> // for std::min

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexCopyLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopySmallIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                    reference_to_const(TensorInfo<T, IndexType>) src,
                                    reference_to_const(TensorInfo<long, IndexType>) indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex < dstCopyDimSize);
#endif
    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);

      dstOffset += dstIndex * dst.strides[dstCopyDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcCopyDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexCopySmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopyLargeIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                    reference_to_const(TensorInfo<T, IndexType>) src,
                                    reference_to_const(TensorInfo<long, IndexType>) indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex < dstCopyDimSize);
#endif
    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstCopyDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcCopyDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexAddLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddSmallIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                   reference_to_const(TensorInfo<T, IndexType>) src,
                                   reference_to_const(TensorInfo<long, IndexType>) indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex < dstAddDimSize);
#endif
    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexAddSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddLargeIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                   reference_to_const(TensorInfo<T, IndexType>) src,
                                   reference_to_const(TensorInfo<long, IndexType>) indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex < dstAddDimSize);
#endif
    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFillLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillSmallIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                    reference_to_const(TensorInfo<long, IndexType>) indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex < dstFillDimSize);
#endif
    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex_ * dst.strides[dstFillDim];

      dst.data[dstOffset] = val;
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFillSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillLargeIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                    reference_to_const(TensorInfo<long, IndexType>) indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(dstIndex_ < dstFillDimSize);
#endif
    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex_ * dst.strides[dstFillDim];

    dst.data[dstOffset] = val;
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                      reference_to_const(TensorInfo<T, IndexType>) src,
                                      reference_to_const(TensorInfo<long, IndexType>) indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(srcIndex < srcSelectDimSize);
#endif
    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectLargeIndex(reference_to_const(TensorInfo<T, IndexType>) dst,
                                      reference_to_const(TensorInfo<T, IndexType>) src,
                                      reference_to_const(TensorInfo<long, IndexType>) indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
#if defined(__NVCC__)
    assert(srcIndex < srcSelectDimSize);
#endif
    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

template <typename IndexType, unsigned int Dims>
struct LinearIndexCalcData {
#if defined(__HIP_PLATFORM_HCC__)
#define MAX_Dim 6
  // sizes for the Tensor dims (from the Tensor, for bounds checking)
  IndexType baseSizes[MAX_Dim];
  // sizes for Tensor dims (either from the Tensor, or the size of the adv indexer at that dim)
  IndexType sizes[MAX_Dim];
  // strides for the Tensor we are indexing into
  IndexType strides[MAX_Dim];
  // these are pointers to the buffers containing the index selected at each dimension
  // for all of the indices we want to generate. If a dimension is not under advanced indexing
  // then the pointer is NULL
  long *advIndexTensors[MAX_Dim];

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    assert(MAX_Dim == 6); // This is hardcoded into the deserialize function signature and the mapping below
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(baseSizes[0]), &baseSizes[i]);
    }
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(sizes[0]), &sizes[i]);
    }
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(strides[0]), &strides[i]);
    }
    for (int i=0; i<MAX_Dim; i++) {
      s.Append(sizeof(advIndexTensors[0]), &advIndexTensors[i]);
    }
  }

  LinearIndexCalcData() {
    for (int i=0; i<MAX_Dim; i++) {
      baseSizes[i] = 0;
      sizes[i] = 0;
      strides[i] = 0;
      advIndexTensors[i] = 0;
    }
  }

  __attribute__((annotate("user_deserialize")))
  LinearIndexCalcData(
                 IndexType baseSizes0, IndexType baseSizes1, IndexType baseSizes2, IndexType baseSizes3, IndexType baseSizes4, IndexType baseSizes5,
                 IndexType sizes0, IndexType sizes1, IndexType sizes2, IndexType sizes3, IndexType sizes4, IndexType sizes5,
                 IndexType strides0, IndexType strides1, IndexType strides2, IndexType strides3, IndexType strides4, IndexType strides5,
                 long*  advIndexTensors0, long*  advIndexTensors1, long*  advIndexTensors2, 
                 long*  advIndexTensors3, long*  advIndexTensors4, long*  advIndexTensors5
                 ) [[cpu]][[hc]] {
    baseSizes[0] = baseSizes0;
    baseSizes[1] = baseSizes1;
    baseSizes[2] = baseSizes2;
    baseSizes[3] = baseSizes3;
    baseSizes[4] = baseSizes4;
    baseSizes[5] = baseSizes5;
    sizes[0] = sizes0;
    sizes[1] = sizes1;
    sizes[2] = sizes2;
    sizes[3] = sizes3;
    sizes[4] = sizes4;
    sizes[5] = sizes5;
    strides[0] = strides0;
    strides[1] = strides1;
    strides[2] = strides2;
    strides[3] = strides3;
    strides[4] = strides4;
    strides[5] = strides5;
    advIndexTensors[0] = advIndexTensors0;
    advIndexTensors[1] = advIndexTensors1;
    advIndexTensors[2] = advIndexTensors2;
    advIndexTensors[3] = advIndexTensors3;
    advIndexTensors[4] = advIndexTensors4;
    advIndexTensors[5] = advIndexTensors5;
  }
#else
  // sizes for the Tensor dims (from the Tensor, for bounds checking)
  IndexType baseSizes[Dims];
  // sizes for Tensor dims (either from the Tensor, or the size of the adv indexer at that dim)
  IndexType sizes[Dims];
  // strides for the Tensor we are indexing into
  IndexType strides[Dims];
  // these are pointers to the buffers containing the index selected at each dimension
  // for all of the indices we want to generate. If a dimension is not under advanced indexing
  // then the pointer is NULL
  long *advIndexTensors[Dims];
#endif
};

template <typename IndexType, unsigned int Dims>
__device__ __forceinline__ long calculateOffset(
  IndexType index,
  LinearIndexCalcData<IndexType, Dims> data
)
{
  IndexType offset = 0;

#pragma unroll
  for (int dim = Dims - 1; dim >= 0; --dim) {
    IndexType sizeAtDim, strideAtDim, indexAtDim, nextIndex;

    strideAtDim = data.strides[dim];
    sizeAtDim = data.sizes[dim];

    if (data.advIndexTensors[dim] != NULL) {
      indexAtDim = data.advIndexTensors[dim][index % sizeAtDim];
      // Check if next dimension is also advanced indexing, if so we must keep the index
      // the same and iterate together
      if (dim > 0 && data.advIndexTensors[dim - 1] != NULL) {
        nextIndex = index;
      } else {
        nextIndex = index / sizeAtDim;
      }
    } else {
      nextIndex = index / sizeAtDim;
      indexAtDim = index - nextIndex * sizeAtDim;
    }

#if defined(__NVCC__)
    assert(indexAtDim < data.baseSizes[dim]);
#endif
    offset += indexAtDim * strideAtDim;
    index = nextIndex;
  }

  return offset;
}

template <typename IndexType, unsigned int Dims>
__global__ void calculateLinearIndices(
  long *output,               // output Tensor for indices
  int elements,               // number of elements in output <-> indices to calculate
  ptrdiff_t baseOffset,       // base offset into the Tensor
  LinearIndexCalcData<IndexType, Dims> data
)
{
  for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < elements;
         i += blockDim.x * gridDim.x) {
      output[i] = baseOffset + calculateOffset<IndexType, Dims>(i, data);
   }
}

#include "generic/THCTensorIndex.cu"
#include "THCGenerateAllTypes.h"
