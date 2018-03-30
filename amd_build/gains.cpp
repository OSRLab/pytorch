sampleMultinomialOnce<real, accreal>
  <<<grid, block,
     requiredShared,
     THCState_getCurrentStream(state)>>>(
  THCudaLongTensor_data(state, self),
  numDist,
  numCategories,
  THCTensor_(data)(state, sampled),
  THCTensor_(data)(state, probDistContig));
THCTensor_(free)(state, sampled);

THCTensor_copyFromDiagonal<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
(THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, size, stride0 + stride1, strideSelf);


    int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
    THCTensor_copyFromDiagonal<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, size, stride0 + stride1, strideSelf);


    renormRowsL1<real>
    <<<grid, block, block.x * sizeof(real),
    THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, t),
                                        rows, cols);

  bitonicSortKVInPlace<real, int64_t, A, -1, GTComp<real>, TYPE, SIZE> \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(         \
      keyInfo,                                                      \
      keySlices,                                                    \
      (TYPE) keySliceSize,                                          \
      (TYPE) keyInfo.strides[collapseKeyDim],                       \
      valueInfo,                                                    \
      (TYPE) valueInfo.strides[collapseValueDim],                   \
      GTComp<real>());
