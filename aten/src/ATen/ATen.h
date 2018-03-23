#pragma once
#if defined(__HIP_PLATFORM_HCC__)
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#endif

#include "ATen/ATenGeneral.h"
#include "ATen/CPUGeneral.h"
#include "ATen/Allocator.h"
#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Generator.h"
#include "ATen/Context.h"
#include "ATen/Storage.h"
#include "ATen/Tensor.h"
#include "ATen/TensorGeometry.h"
#include "ATen/Functions.h"
#include "ATen/Formatting.h"
#include "ATen/TensorOperators.h"
#include "ATen/TensorMethods.h"
#include "ATen/Dispatch.h"
