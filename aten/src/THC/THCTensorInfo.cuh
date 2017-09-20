#ifndef THC_TENSOR_INFO_INC
#define THC_TENSOR_INFO_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCIntegerDivider.cuh"
#include "THCTensor.h"

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 25

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CUTORCH_STR(X) #X
#define CUTORCH_DIM_WARNING "tensor too large or too many (>" \
  CUTORCH_STR(MAX_CUTORCH_DIMS) ") dimensions"

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_CUTORCH_DIMS],
             IndexType st[MAX_CUTORCH_DIMS]);

#ifdef __HCC__
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    assert(MAX_CUTORCH_DIMS == 25); // This is hardcoded into the deserialize function signature and the mapping below
    s.Append(sizeof(data), &data);
    for (int i=0; i<MAX_CUTORCH_DIMS; i++) {
      s.Append(sizeof(sizes[0]), &sizes[i]);
    }
    for (int i=0; i<MAX_CUTORCH_DIMS; i++) {
      s.Append(sizeof(strides[0]), &strides[i]);
    }
    s.Append(sizeof(dims), &dims);
  }

  __attribute__((annotate("user_deserialize")))
  TensorInfo(T* p, 
             IndexType sz00, IndexType sz01, IndexType sz02, IndexType sz03, IndexType sz04,
             IndexType sz05, IndexType sz06, IndexType sz07, IndexType sz08, IndexType sz09,
             IndexType sz10, IndexType sz11, IndexType sz12, IndexType sz13, IndexType sz14,
             IndexType sz15, IndexType sz16, IndexType sz17, IndexType sz18, IndexType sz19,
             IndexType sz20, IndexType sz21, IndexType sz22, IndexType sz23, IndexType sz24,
             IndexType str00, IndexType str01, IndexType str02, IndexType str03, IndexType str04,
             IndexType str05, IndexType str06, IndexType str07, IndexType str08, IndexType str09,
             IndexType str10, IndexType str11, IndexType str12, IndexType str13, IndexType str14,
             IndexType str15, IndexType str16, IndexType str17, IndexType str18, IndexType str19,
             IndexType str20, IndexType str21, IndexType str22, IndexType str23, IndexType str24,
             int d) [[cpu]][[hc]] {
    data = p;
    sizes[ 0] = sz00; sizes[ 1] = sz01; sizes[ 2] = sz02; sizes[ 3] = sz03; sizes[ 4] = sz04;
    sizes[ 5] = sz05; sizes[ 6] = sz06; sizes[ 7] = sz07; sizes[ 8] = sz08; sizes[ 9] = sz09;
    sizes[10] = sz10; sizes[11] = sz11; sizes[12] = sz12; sizes[13] = sz13; sizes[14] = sz14;
    sizes[15] = sz15; sizes[16] = sz16; sizes[17] = sz17; sizes[18] = sz18; sizes[19] = sz19;
    sizes[20] = sz20; sizes[21] = sz21; sizes[22] = sz22; sizes[23] = sz23; sizes[24] = sz24;
    strides[ 0] = str00; strides[ 1] = str01; strides[ 2] = str02; strides[ 3] = str03; strides[ 4] = str04;
    strides[ 5] = str05; strides[ 6] = str06; strides[ 7] = str07; strides[ 8] = str08; strides[ 9] = str09;
    strides[10] = str10; strides[11] = str11; strides[12] = str12; strides[13] = str13; strides[14] = str14;
    strides[15] = str15; strides[16] = str16; strides[17] = str17; strides[18] = str18; strides[19] = str19;
    strides[20] = str20; strides[21] = str21; strides[22] = str22; strides[23] = str23; strides[24] = str24;
    dims = d;
  }
#endif

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

  // Collapses all runs of successive dimensions if the size/strides
  // match up within the run and there are no holes between the
  // dimensions.
  // If excludeDim is set (not -1), then excludeDim will not be
  // collapsed with any other dimension.
  // Function returns the new dimension index that excludeDim maps to,
  // since the collapsed dimensions are <= the input dimensions.
  int collapseDims(int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  __host__ __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  T* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_CUTORCH_DIMS],
                                     IndexType st[MAX_CUTORCH_DIMS]) {
  data = p;
  dims = dim;
  assert(dims > 0 && dims < MAX_CUTORCH_DIMS);

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename T, typename IndexType>
void
TensorInfo<T, IndexType>::reduceDim(int dim) {
  assert(dim < dims && dim >= 0);
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int
TensorInfo<T, IndexType>::collapseDims(int excludeDim) {
  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = dims - 1; i >= 0; --i) {
    if (i == excludeDim) {
      // We cannot collapse this dimension, even if it is size 1
      firstNonOneDim = i;
      break;
    }

    if (sizes[i] != 1) {
      firstNonOneDim = i;
      break;
    }
  }

  // Special case: if all dimensions are of size 1, then this is a
  // single-point tensor that we still have to operate on. Reduce to a
  // single point.
  if (firstNonOneDim == -1) {
    assert(excludeDim == -1);

    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    // Everything effectively got collapsed into this dimension
    return 0;
  }

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Skip the leading size 1 dims
  numCollapsed += dims - 1 - firstNonOneDim;

  // We perform one pass through to determine how many dimensions we
  // can collapse, before calculating the actual size of the collapsed
  // dimensions.
  // size/strideInner are the size/strides of the previous inner
  // non-collapsible dim we encounter.
  int64_t sizeInner = sizes[firstNonOneDim];
  int64_t strideInner = strides[firstNonOneDim];

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    int64_t sizeOuter = sizes[i];
    int64_t strideOuter = strides[i];

    // Don't collapse this dimension if we want to exclude it from
    // collapsing.
    // Since this code is attempting to collapse a subsequent
    // dimension (i) with the preceding dimension (i + 1), we can only
    // perform collapsing if the preceding dimension can be collapsed
    // (i.e., not excludeDim)
    if ((excludeDim != i) && (excludeDim != i + 1)) {
      // The next outermost dimension can be skipped if size 1
      if (sizeOuter == 1) {
        ++numCollapsed;
        continue;
      }

      // If the next outermost dimension is contiguous with the
      // previous non-collapsed one, collapse it
      if (strideOuter == strideInner * sizeInner) {
        ++numCollapsed;

        // This is the run of collapsed dimensions' size
        sizeInner = sizeInner * sizeOuter;
        continue;
      }
    }

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // because it is excluded from collapsing, or it is not contiguous
    // with the previous inner dimension.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  // This will be our new size/stride and dimension.
  IndexType newSizes[MAX_CUTORCH_DIMS];
  IndexType newStrides[MAX_CUTORCH_DIMS];

  assert(numCollapsed < dims);
  int newDims = dims - numCollapsed;

  // We return the index of the excluded dimension that is excluded
  // from being collapsed here.
  int returnDim = -1;

  // We perform a second pass through the dimensions to actually
  // calculate the size of the collapsed dimensions.
  int collapsedIndex = dims - numCollapsed - 1;
  newSizes[collapsedIndex] = sizes[firstNonOneDim];
  newStrides[collapsedIndex] = strides[firstNonOneDim];

  if (firstNonOneDim == excludeDim) {
    returnDim = collapsedIndex;
  }

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    IndexType sizeOuter = sizes[i];
    IndexType strideOuter = strides[i];

    if ((excludeDim != i) && (excludeDim != i + 1)) {
      if (sizeOuter == 1) {
        // skip
        continue;
      }

      if (strideOuter == newSizes[collapsedIndex] * newStrides[collapsedIndex]) {
        // collapse
        newSizes[collapsedIndex] *= sizeOuter;
        continue;
      }
    }

    // Otherwise, strides don't match, or dim `i` is excluded from
    // collapsing.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    assert(collapsedIndex < newDims);
    newSizes[collapsedIndex] = sizeOuter;
    newStrides[collapsedIndex] = strideOuter;

    if (excludeDim == i) {
      returnDim = collapsedIndex;
    }
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);
  assert((excludeDim == -1) || (returnDim != -1));

  dims = newDims;

  for (int i = 0; i < dims; ++i) {
    sizes[i] = newSizes[i];
    strides[i] = newStrides[i];
  }

  // After collapsing, the original `excludeDim` may have been
  // renumbered to this new `returnDim`, since some dimensions could
  // have been collapsed.
  return returnDim;
}

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }
    offset += linearId * info.strides[0];

    return offset;
  }
};

// For contiguous tensors, the offset = index
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -2> {
  static inline __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    return linearId;
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }
    offset += linearId * info.strides[0];

    return offset;
  }
};

// OffsetInfo is a faster implementation of IndexToOffset that uses faster
// integer division: we transform each division into integer multiplication by a
// pre-computed constant.  (See IntDivider for details.)
template <typename T, typename IndexType, int Dims>
struct OffsetInfo {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo) {
    assert(tinfo.dims == Dims);
    data = tinfo.data;

    for (int i = 0; i < Dims; ++i) {
      sizes[i] = IntDivider<IndexType>(tinfo.sizes[i]);
      strides[i] = tinfo.strides[i];
    }
  }

  __host__ __device__ T* get(IndexType linearIndex) const {
    IndexType offset = 0;

    for (int i = Dims - 1; i > 0; --i) {
      DivMod<IndexType> divmod = sizes[i].divmod(linearIndex);
      linearIndex = divmod.div;
      offset += divmod.mod * strides[i];
    }

    offset += linearIndex * strides[0];
    return &data[offset];
  }

  T* data;
  IntDivider<IndexType> sizes[Dims];
  IndexType strides[Dims];
};

// For contiguous tensors (Dims=-2), offset equals linear index.
template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -2> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
    : data(tinfo.data) {
    assert(tinfo.isContiguous());
  }

  __host__ __device__ T* get(IndexType linearIndex) const {
    return &data[linearIndex];
  }

  T* data;
};

// Dims=-1 is used when the dimension is unknown at compile time.
//
// Unfortunately, pre-computation does not work here, because of a bug in nvcc
// (tested on CUDA 8.0): if a kernel argument contains an array that is
// dynamically accessed, the whole array is first copied into the local memory.
// (That is, every kernel thread makes its own copy of the argument, even if it
// is never updated.)  Pre-computation makes it worse because now we have more
// data to copy.
//
// So let's fall back to vanilla division approach.

template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -1> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
    : tinfo(tinfo) { }

  __host__ __device__ T* get(IndexType linearIndex) const {
    IndexType offset = IndexToOffset<T, IndexType, -1>::get(linearIndex, tinfo);
    return &tinfo.data[offset];
  }

  TensorInfo<T, IndexType> tinfo;
};

#endif // THC_TENSOR_INFO_INC
