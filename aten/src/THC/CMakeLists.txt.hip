# HIP_PATH
IF(NOT DEFINED $ENV{HIP_PATH})
  SET(HIP_PATH /opt/rocm/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

# HCC_PATH
IF(NOT DEFINED $ENV{HCC_PATH})
  SET(HCC_PATH /opt/rocm/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()

# HIPBLAS_PATH
IF(NOT DEFINED $ENV{HIPBLAS_PATH})
  SET(HIPBLAS_PATH /opt/rocm/hipblas)
ELSE()
  SET(HIPBLAS_PATH $ENV{HIPBLAS_PATH})
ENDIF()

# HIPRNG_PATH
IF(NOT DEFINED $ENV{HIPRNG_PATH})
  SET(HIPRNG_PATH /opt/rocm/hcrng)
ELSE()
  SET(HIPRNG_PATH $ENV{HIPRNG_PATH})
ENDIF()

# HIPSPARSE_PATH
IF(NOT DEFINED $ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH /opt/rocm/hcsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()

SET(THRUST_PATH "/root/Thrust")

# load HIP cmake module and load platform id
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
FIND_PACKAGE(HIP 1.0 REQUIRED)
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE PLATFORM)
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

SET(HIP_CXX_FLAGS "-D__HIP_PLATFORM_HCC__ -I/opt/rocm/hip/include -I/opt/rocm/hcc/include" ${HIP_CXX_FLAGS})

SET(HIP_HIPCC_FLAGS "-DGENERIC_GRID_LAUNCH=1 ${CMAKE_CXX_FLAGS}")
SET(HIP_HIPCC_FLAGS "-DGENERIC_GRID_LAUNCH=1 ${HIP_HIPCC_FLAGS}")
SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
SET(CMAKE_C_COMPILER ${HIP_HIPCC_EXECUTABLE})
SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS  "-std=c++11 ${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

FIND_PACKAGE(HIP 1.0 REQUIRED)

MESSAGE(STATUS "ROCM TRUE:")
MESSAGE(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})

INCLUDE_DIRECTORIES(${HIPBLAS_PATH}/include)
INCLUDE_DIRECTORIES(${HIPSPARSE_PATH}/include)
INCLUDE_DIRECTORIES(${HIPRNG_PATH}/include)
INCLUDE_DIRECTORIES(${THRUST_PATH})

set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

set(extra_src)
# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
              "#include \"../THC${THC_FILE}.cuh\"\n#include \"../generic/THC${THC_FILE}.cu\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
      endif()
      LIST(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

IF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/THCHalf.cu)
ENDIF()

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingHostAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorRandom.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCThreadLocal.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCReduceApplyUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCBlas.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMath.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathBlas.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathPairwise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathReduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathScan.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorIndex.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorConv.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorRandom.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorScatterGather.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorTopK.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorSort.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorTypeUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSortUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMode.cu
  ${extra_src}
  PARENT_SCOPE)

INSTALL(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCBlas.h
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCStream.h
          THCThreadLocal.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceAll.cuh
          THCReduceApplyUtils.cuh
          THCTensorMathReduce.cuh
          THCAsmUtils.cuh
          THCAtomics.cuh
          THCScanUtils.cuh
          THCSortUtils.cuh
          THCAllocator.h
          THCCachingAllocator.h
          THCCachingHostAllocator.h
          THCDeviceUtils.cuh
          THCDeviceTensor.cuh
          THCDeviceTensor-inl.cuh
          THCDeviceTensorUtils.cuh
          THCDeviceTensorUtils-inl.cuh
          THCGenerateAllTypes.h
          THCGenerateByteType.h
          THCGenerateCharType.h
          THCGenerateShortType.h
          THCGenerateIntType.h
          THCGenerateLongType.h
          THCGenerateHalfType.h
          THCGenerateFloatType.h
          THCGenerateFloatTypes.h
          THCGenerateDoubleType.h
          THCHalf.h
          THCIntegerDivider.cuh
          THCNumerics.cuh
          THCTensorSort.cuh
          THCTensorInfo.cuh
          THCTensorMathPointwise.cuh
          THCTensorTypeUtils.cuh
          THCTensorRandom.cuh
          THCTensorMathMagma.cuh
          THCThrustAllocator.cuh
          THCTensorMode.cuh
          THCTensorTopK.cuh
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

INSTALL(FILES
          generic/THCStorage.c
          generic/THCStorage.cu
          generic/THCStorage.h
          generic/THCTensor.c
          generic/THCTensor.cu
          generic/THCTensor.h
          generic/THCStorageCopy.c
          generic/THCStorageCopy.cu
          generic/THCStorageCopy.h
          generic/THCTensorCopy.c
          generic/THCTensorCopy.cu
          generic/THCTensorCopy.h
          generic/THCTensorMasked.h
          generic/THCTensorMasked.cu
          generic/THCTensorMath.h
          generic/THCTensorMath.cu
          generic/THCTensorMathBlas.cu
          generic/THCTensorMathBlas.h
          generic/THCTensorMathCompare.h
          generic/THCTensorMathCompare.cu
          generic/THCTensorMathCompareT.h
          generic/THCTensorMathCompareT.cu
          generic/THCTensorMathMagma.h
          generic/THCTensorMathMagma.cu
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.cu
          generic/THCTensorMathPointwise.h
          generic/THCTensorMathPointwise.cu
          generic/THCTensorMathReduce.h
          generic/THCTensorMathReduce.cu
          generic/THCTensorMathScan.h
          generic/THCTensorMathScan.cu
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.cu
          generic/THCTensorIndex.h
          generic/THCTensorIndex.cu
          generic/THCTensorSort.h
          generic/THCTensorSort.cu
          generic/THCDeviceTensorUtils.cu
          generic/THCTensorRandom.h
          generic/THCTensorRandom.cu
          generic/THCTensorMode.h
          generic/THCTensorMode.cu
          generic/THCTensorTopK.h
          generic/THCTensorTopK.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
