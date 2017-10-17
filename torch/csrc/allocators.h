#pragma once

#include <Python.h>
#include <type_traits>
#include <memory>

#include <TH/TH.h>
#if defined(WITH_CUDA) || defined(WITH_ROCM)
#include <THC/THC.h>
#endif

#include "torch/csrc/utils/object_ptr.h"

// Adapted from fblualib
class ObjectPtrAllocator {
public:
  ObjectPtrAllocator(PyObject *wrapped_object):
      ObjectPtrAllocator(wrapped_object, &THDefaultAllocator, nullptr) {}

  ObjectPtrAllocator(PyObject *wrapped_object, THAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

  void* malloc(ptrdiff_t size);
  void* realloc(void* ptr, ptrdiff_t size);
  void free(void* ptr);

  THPObjectPtr object;
  THAllocator *allocator;
  void *allocatorContext;
};

class StorageWeakRefAllocator: public ObjectPtrAllocator {
public:
  StorageWeakRefAllocator(PyObject *wrapped_object, THAllocator *alloc, void *ctx):
    ObjectPtrAllocator(wrapped_object, alloc, ctx) {}

  void free(void* ptr);
};

#if defined(WITH_CUDA) || defined(WITH_ROCM)
class CudaStorageWeakRefAllocator {
public:
  CudaStorageWeakRefAllocator(PyObject *wrapped_object, THCDeviceAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

#if defined(__HIP_PLATFORM_HCC__)
  hipError_t malloc(void** ptr, size_t size, hipStream_t stream);
  hipError_t realloc(void** ptr, size_t old_size, size_t size, hipStream_t stream);
  hipError_t free(void* ptr);
#else
  cudaError_t malloc(void** ptr, size_t size, cudaStream_t stream);
  cudaError_t realloc(void** ptr, size_t old_size, size_t size, cudaStream_t stream);
  cudaError_t free(void* ptr);
#endif

  THPObjectPtr object;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
};
#endif

extern THAllocator THObjectPtrAllocator;
extern THAllocator THStorageWeakRefAllocator;
#if defined(WITH_CUDA) || defined(WITH_ROCM)
extern THCDeviceAllocator THCStorageWeakRefAllocator;
#endif
