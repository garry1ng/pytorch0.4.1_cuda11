#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SparseLinear.cu"
#else

static bool THNN_(checkInput)(THCTensor* t)
{
  return !t->is_empty() && t->_dim() == 2 && t->size[1] == 3;
}

static bool THNN_(checkSize2D)(THCTensor* t, int64_t size0, int64_t size1)
{
  return !t->is_empty() && t->_dim() == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool THNN_(checkSize1D)(THCTensor* t, int64_t size0)
{
  return !t->is_empty() && t->_dim() == 1 && t->size[0] == size0;
}

static inline void THNN_(copyCudaFloatingType)(THCState *state, THCudaIntTensor *buf, THCTensor *t) {
  #ifdef THC_REAL_IS_FLOAT
  THCudaIntTensor_copyCudaFloat(state, buf, t);
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaIntTensor_copyCudaDouble(state, buf, t);
  #elif defined(THC_REAL_IS_HALF)
  THCudaIntTensor_copyCudaHalf(state, buf, t);
  #endif
}

// TODO: add
// #if (CUDART_VERSION >= 11000)
// template<class MatType, class IndType>
// inline void
// generic_SpMM_SL(cusparseHandle_t handle, cusparseOperation_t transA,
//              int m, int n, int k, int nnz,
//              const MatType *alpha,
//              cusparseMatDescr_t dummy_descr,
//              const MatType *Avals,
//              const IndType *rowPtr,
//              const IndType *colInd,
//              const MatType *Bvals,
//              int ldb, 
//              const MatType *beta,
//              MatType *Cvals,
//              int ldc,
//              cudaDataType matType)
// {
//     // Create the matrix descriptors
//     cusparseSpMatDescr_t matA_descr;
//     cusparseDnMatDescr_t matB_descr;
//     cusparseDnMatDescr_t matC_descr;
//     cusparseCreateCsr(&matA_descr, m, k, nnz, const_cast<IndType*>(rowPtr), const_cast<IndType*>(colInd),
//                           const_cast<MatType*>(Avals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType);
//         cusparseCreateDnMat(&matB_descr, k, n, ldb, const_cast<MatType*>(Bvals), matType, CUSPARSE_ORDER_COL);
//         cusparseCreateDnMat(&matC_descr, m, n, ldc, const_cast<MatType*>(Cvals), matType, CUSPARSE_ORDER_COL);

//     // Check if a buffer is required, and if so allocate it using caching allocator
//     size_t bufferSize = 0;
//         cusparseSpMM_bufferSize(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr,
//                                 beta, matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

//     void* dBuffer = NULL;
//     if(bufferSize > 0)
//     {
//         cudaMalloc(&dBuffer, bufferSize);
//     }

//     // Compute the sparse matrix - dense matrix product
//         cusparseSpMM(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr, beta,
//                      matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

//     // Clean up
//     cusparseDestroySpMat(matA_descr);
//     cusparseDestroyDnMat(matB_descr);
//     cusparseDestroyDnMat(matC_descr);

//     // if(bufferSize > 0)
//     // {
//     //     cudaFreeAsync(dBuffer);
//     // }
// }

// static inline void generic_SpMM_SL(cusparseHandle_t handle, cusparseOperation_t transA, 
//              int m, int n, int k, int nnz,
//               float *alpha,
//              cusparseMatDescr_t dummy_descr,
//             float *Avals,
//             int32_t *rowPtr,
//             int32_t *colInd,
//             float *Bvals,
//              int ldb, 
//             float *beta,
//              float *Cvals,
//              int ldc,
//              cudaDataType matType)
// {
//     // Create the matrix descriptors
//     cusparseSpMatDescr_t matA_descr;
//     cusparseDnMatDescr_t matB_descr;
//     cusparseDnMatDescr_t matC_descr;
//     cusparseCreateCsr(&matA_descr, m, k, nnz, (rowPtr), (colInd),
//                           (Avals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType);
//     cusparseCreateDnMat(&matB_descr, k, n, ldb, (Bvals), matType, CUSPARSE_ORDER_COL);
//     cusparseCreateDnMat(&matC_descr, m, n, ldc, (Cvals), matType, CUSPARSE_ORDER_COL);

//     // Check if a buffer is required, and if so allocate it using caching allocator
//     size_t bufferSize = 0;
//     cusparseSpMM_bufferSize(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr,
//                                 beta, matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

//     void* dBuffer = NULL;
//     if(bufferSize > 0)
//     {
//         cudaMalloc(&dBuffer, bufferSize);
//     }

//     // Compute the sparse matrix - dense matrix product
//     cusparseSpMM(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr, beta,
//                      matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

//     // Clean up
//     cusparseDestroySpMat(matA_descr);
//     cusparseDestroyDnMat(matB_descr);
//     cusparseDestroyDnMat(matC_descr);

//     // if(bufferSize > 0)
//     // {
//     //     cudaFreeAsync(dBuffer);
//     // }
// }

// static inline void generic_SpMM_SL(cusparseHandle_t handle, cusparseOperation_t transA, 
//              int m, int n, int k, int nnz,
//               double *alpha,
//              cusparseMatDescr_t dummy_descr,
//             double *Avals,
//             int32_t *rowPtr,
//             int32_t *colInd,
//             double *Bvals,
//              int ldb, 
//             double *beta,
//              double *Cvals,
//              int ldc,
//              cudaDataType matType)
// {
//     // Create the matrix descriptors
//     cusparseSpMatDescr_t matA_descr;
//     cusparseDnMatDescr_t matB_descr;
//     cusparseDnMatDescr_t matC_descr;
//     cusparseCreateCsr(&matA_descr, m, k, nnz, (rowPtr), (colInd),
//                           (Avals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType);
//     cusparseCreateDnMat(&matB_descr, k, n, ldb, (Bvals), matType, CUSPARSE_ORDER_COL);
//     cusparseCreateDnMat(&matC_descr, m, n, ldc, (Cvals), matType, CUSPARSE_ORDER_COL);

//     // Check if a buffer is required, and if so allocate it using caching allocator
//     size_t bufferSize = 0;
//     cusparseSpMM_bufferSize(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr,
//                                 beta, matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

//     void* dBuffer = NULL;
//     if(bufferSize > 0)
//     {
//         cudaMalloc(&dBuffer, bufferSize);
//     }

//     // Compute the sparse matrix - dense matrix product
//     cusparseSpMM(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr, beta,
//                      matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

//     // Clean up
//     cusparseDestroySpMat(matA_descr);
//     cusparseDestroyDnMat(matB_descr);
//     cusparseDestroyDnMat(matC_descr);

//     // if(bufferSize > 0)
//     // {
//     //     cudaFreeAsync(dBuffer);
//     // }
// }

// #endif

void THNN_(SparseLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias)
{
  THAssert(THCTensor_(checkGPU)(state, 4, input, output, weight, bias));

  int64_t h;
  int64_t outDim = THCTensor_(size)(state, weight, 0);
  int64_t inDim = THCTensor_(size)(state, weight, 1);

  THArgCheck(THNN_(checkInput)(input), 2, "input size must be nnz x 3");
  AT_CHECK(!output->is_empty() && THCTensor_(nDimension)(state, output) == 2,
           "output must be batchsize x outputsize, got size: ", output->sizes());
  THArgCheck(THNN_(checkSize1D)(bias, outDim), 5, "bias size wrong");

  weight = THCTensor_(newContiguous)(state, weight);
  
  int64_t batchnum = THCTensor_(size)(state, output, 0);
  int64_t nnz = THCTensor_(size)(state, input, 0);

  THCTensor *buffer = THCTensor_(new)(state);
  THCTensor *sel = THCTensor_(new)(state);
  THCTensor *values = THCTensor_(new)(state);
  THCudaIntTensor *rowbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *csrPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *colInds = THCudaIntTensor_new(state);

  THCTensor_(resize1d)(state, values, nnz);
  THCudaIntTensor_resize1d(state, rowbuf, nnz);
  THCudaIntTensor_resize1d(state, colInds, nnz);
  THCudaIntTensor_resize1d(state, csrPtrs, batchnum+1);

  // Get data ready for cusparse, need CudaInt buffers
  // We do not need to sort, since rows are already in order
  // If rows might get out of order in future implementations, or if cusparse
  //    complains with an illegal memory access, sort like we do in AccGradParameters
  THCTensor_(select)(state, sel, input, 1, 0);
  THNN_(copyCudaFloatingType)(state, rowbuf, sel);
  THCTensor_(select)(state, sel, input, 1, 1);
  THNN_(copyCudaFloatingType)(state, colInds, sel);
  THCTensor_(select)(state, sel, input, 1, 2);
  THCTensor_(copyCuda)(state, values, sel);

  init_cusparse();
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, rowbuf), nnz, batchnum,
      THCudaIntTensor_data(state, csrPtrs), CUSPARSE_INDEX_BASE_ONE);

  // output = bias
  THCTensor_(resize2d)(state, buffer, outDim, batchnum);
  THCTensor_(zero)(state, buffer);
  for (h=0; h<batchnum; h++) {
    THCTensor_(select)(state, sel, buffer, 1, h);
    THCTensor_(copy)(state, sel, bias);
  }

  // output = W * x
  real one = ScalarConvert<int, real>::to(1);
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);
  /*
  #if (CUDART_VERSION >= 11000)
  #ifdef THC_REAL_IS_FLOAT
  generic_SpMM_SL(cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      batchnum, outDim, inDim, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, csrPtrs),
      THCudaIntTensor_data(state, colInds),
      THCTensor_(data)(state, weight), inDim,
      &one, THCTensor_(data)(state, buffer), batchnum,
      CUDA_R_32F);
  #else
  generic_SpMM_SL(cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      batchnum, outDim, inDim, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, csrPtrs),
      THCudaIntTensor_data(state, colInds),
      THCTensor_(data)(state, weight), inDim,
      &one, THCTensor_(data)(state, buffer), batchnum,
      CUDA_R_64F);
  #endif
  #endif
  
  #if !(CUDART_VERSION >= 11000)
  #ifdef THC_REAL_IS_FLOAT
  cusparseScsrmm(cusparse_handle,
  #elif defined(THC_REAL_IS_DOUBLE)
  cusparseDcsrmm(cusparse_handle,
  #endif
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      batchnum, outDim, inDim, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, csrPtrs),
      THCudaIntTensor_data(state, colInds),
      THCTensor_(data)(state, weight), inDim,
      &one, THCTensor_(data)(state, buffer), batchnum
  );
  #endif
  */
  THCTensor_(transpose)(state, buffer, NULL, 0, 1);

  // We do work in the buffer to keep the output contiguous
  THCTensor_(copy)(state, output, buffer);

  cusparseDestroyMatDescr(descr);
  descr = 0;
  THCTensor_(free)(state, buffer);
  THCTensor_(free)(state, sel);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, weight);
  THCudaIntTensor_free(state, rowbuf);
  THCudaIntTensor_free(state, colInds);
  THCudaIntTensor_free(state, csrPtrs);
}

void THNN_(SparseLinear_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale)
{
  int64_t outDim = THCTensor_(size)(state, weight, 0);
  int64_t inDim = THCTensor_(size)(state, weight, 1);

  THArgCheck(THNN_(checkInput)(input), 2, "input size must be batchsize x nnz x 2");
  THArgCheck(THNN_(checkSize2D)(gradWeight, outDim, inDim), 4, "gradWeight size wrong");
  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 5, "gradBias size wrong");

  weight = THCTensor_(newContiguous)(state, weight);
  int64_t nnz = THCTensor_(size)(state, input, 0);
  int64_t batchnum = THCTensor_(size)(state, gradOutput, 0);

  THCTensor *buf = THCTensor_(new)(state);
  THCTensor *cols = THCTensor_(new)(state);
  THCTensor *sel = THCTensor_(new)(state);
  THCudaLongTensor *inds = THCudaLongTensor_new(state);
  THCTensor *values = THCTensor_(new)(state);
  THCudaIntTensor *colbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *colPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *rowInds = THCudaIntTensor_new(state);

  THCTensor_(select)(state, sel, input, 1, 0); // rowInds
  THCTensor_(select)(state, cols, input, 1, 1); // colInds
  THCTensor_(cadd)(state, buf, sel, batchnum, cols); // colInds * buatchdim + rowInds
  THCTensor_(sort)(state, buf, inds, buf, 0, 0); // Indices are now in ind
  THCTensor_(indexSelect)(state, buf, input, 0, inds);

  THCTensor_(resize1d)(state, values, nnz);
  THCudaIntTensor_resize1d(state, colbuf, nnz);
  THCudaIntTensor_resize1d(state, rowInds, nnz);
  THCudaIntTensor_resize1d(state, colPtrs, inDim+1);

  // Get data ready for cusparse, need CudaInt buffers
  THCTensor_(select)(state, sel, buf, 1, 0);
  THNN_(copyCudaFloatingType)(state, rowInds, sel);
  THCTensor_(select)(state, sel, buf, 1, 1);
  THNN_(copyCudaFloatingType)(state, colbuf, sel);
  THCTensor_(select)(state, sel, buf, 1, 2);
  THCTensor_(copyCuda)(state, values, sel);

  init_cusparse();
  // Secretly coo2csc
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, colbuf), nnz, inDim,
      THCudaIntTensor_data(state, colPtrs), CUSPARSE_INDEX_BASE_ONE);

  // FORTRAN expects contiguous col-major matricies
  THCTensor *tgradOutput = THCTensor_(new)(state);
  THCTensor_(transpose)(state, tgradOutput, gradOutput, 0, 1);
  THCTensor_(resize2d)(state, buf, batchnum, outDim);
  THCTensor_(copy)(state, buf, tgradOutput);
  THCTensor_(free)(state, tgradOutput);

  real one = ScalarConvert<int, real>::to(1);
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);
  /*
  #if (CUDART_VERSION >= 11000)
  #ifdef THC_REAL_IS_FLOAT
  generic_SpMM_SL(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      inDim, outDim, batchnum, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, colPtrs),
      THCudaIntTensor_data(state, rowInds),
      THCTensor_(data)(state, buf), batchnum,
      &one, THCTensor_(data)(state, gradWeight), inDim,
      CUDA_R_32F);
  #else
  generic_SpMM_SL(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      inDim, outDim, batchnum, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, colPtrs),
      THCudaIntTensor_data(state, rowInds),
      THCTensor_(data)(state, buf), batchnum,
      &one, THCTensor_(data)(state, gradWeight), inDim,
      CUDA_R_64F);
  #endif
  #endif
  #if !(CUDART_VERSION >= 11000)
  #ifdef THC_REAL_IS_FLOAT
  cusparseScsrmm(cusparse_handle,
  #elif defined(THC_REAL_IS_DOUBLE)
  cusparseDcsrmm(cusparse_handle,
  #endif
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      inDim, outDim, batchnum, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, colPtrs),
      THCudaIntTensor_data(state, rowInds),
      THCTensor_(data)(state, buf), batchnum,
      &one, THCTensor_(data)(state, gradWeight), inDim
  );
  #endif
  */
  
  THCTensor_(sum)(state, buf, gradOutput, 0, 1);
  THCTensor_(resize1d)(state, buf, outDim);
  THCTensor_(cadd)(state, gradBias, gradBias, scale, buf);

  if (weightDecay != 0)
  {
    THCTensor_(cadd)(state, gradWeight, gradWeight, weightDecay, weight);
    THCTensor_(cadd)(state, gradBias, gradBias, weightDecay, bias);
  }

  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, buf);
  THCTensor_(free)(state, sel);
  THCTensor_(free)(state, cols);
  THCudaLongTensor_free(state, inds);
  THCTensor_(free)(state, values);
  THCudaIntTensor_free(state, colbuf);
  THCudaIntTensor_free(state, rowInds);
  THCudaIntTensor_free(state, colPtrs);
}

void THNN_(SparseLinear_legacyUpdateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}
void THNN_(SparseLinear_legacyAccGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}

// Dense updates are pretty fast on the GPU
void THNN_(SparseLinear_zeroGradParameters)(
           THCState *state,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput) {
  THCTensor_(zero)(state, gradWeight);
  THCTensor_(zero)(state, gradBias);
}

void THNN_(SparseLinear_updateParameters)(
           THCState *state,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput,
           accreal learningRate) {
  THCTensor_(cadd)(state, weight, weight, -learningRate, gradWeight);
  THCTensor_(cadd)(state, bias, bias, -learningRate, gradBias);
}

#endif