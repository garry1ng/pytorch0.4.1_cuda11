#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <ATen/Error.h>
#include <ATen/Context.h>

#include <TH/THGeneral.h>

#include <cusparse.h>

namespace at { namespace native { namespace sparse { namespace cuda {

#ifndef __HIP_PLATFORM_HCC__

// std::string cusparseGetErrorString(cusparseStatus_t status) {
//   switch(status)
//   {
//     case CUSPARSE_STATUS_SUCCESS:
//       return "success";

//     case CUSPARSE_STATUS_NOT_INITIALIZED:
//       return "library not initialized";

//     case CUSPARSE_STATUS_ALLOC_FAILED:
//       return "resource allocation failed";

//     case CUSPARSE_STATUS_INVALID_VALUE:
//       return "an invalid numeric value was used as an argument";

//     case CUSPARSE_STATUS_ARCH_MISMATCH:
//       return "an absent device architectural feature is required";

//     case CUSPARSE_STATUS_MAPPING_ERROR:
//       return "an access to GPU memory space failed";

//     case CUSPARSE_STATUS_EXECUTION_FAILED:
//       return "the GPU program failed to execute";

//     case CUSPARSE_STATUS_INTERNAL_ERROR:
//       return "an internal operation failed";

//     case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
//       return "the matrix type is not supported by this function";

//     case CUSPARSE_STATUS_ZERO_PIVOT:
//       return "an entry of the matrix is either structural zero or numerical zero (singular block)";

//     default:
//       {
//         std::ostringstream oss;
//         oss << "unknown error " << static_cast<int64_t>(status);
//         return oss.str();
//       }
//   }
// }

inline void CUSPARSE_CHECK(cusparseStatus_t status)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    AT_ERROR("cusparse runtime error: ", cusparseGetErrorString(status));
  }
}

inline cusparseHandle_t setCUDASparseStream() {
  cusparseHandle_t handle = globalContext().getCurrentCUDASparseHandle();
  cusparseSetStream(handle, globalContext().getCurrentCUDAStream());
  return handle;
}

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  AT_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);
  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, coorowind, nnz, m, csrrowptr,
    TH_INDEX_BASE ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO
  ));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    AT_ERROR("trans must be one of: t, n, c");
  }
}

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

#if (CUDART_VERSION >= 11000)
template<class MatType, class IndType>
inline void
generic_SpMM2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB,
             int m, int n, int k, int nnz,
             const MatType *alpha,
             cusparseMatDescr_t dummy_descr,
             const MatType *Avals,
             const IndType *rowPtr,
             const IndType *colInd,
             const MatType *Bvals,
             int ldb, 
             const MatType *beta,
             MatType *Cvals,
             int ldc,
             cudaDataType matType)
{
    // Create the matrix descriptors
    cusparseSpMatDescr_t matA_descr;
    cusparseDnMatDescr_t matB_descr;
    cusparseDnMatDescr_t matC_descr;
    CUSPARSE_CHECK(
        cusparseCreateCsr(&matA_descr, m, k, nnz, const_cast<IndType*>(rowPtr), const_cast<IndType*>(colInd),
                          const_cast<MatType*>(Avals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType));
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&matB_descr, k, n, ldb, const_cast<MatType*>(Bvals), matType, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(
        cusparseCreateDnMat(&matC_descr, m, n, ldc, const_cast<MatType*>(Cvals), matType, CUSPARSE_ORDER_COL));

    // Check if a buffer is required, and if so allocate it using caching allocator
    size_t bufferSize = 0;
    CUSPARSE_CHECK(
        cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA_descr, matB_descr,
                                beta, matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    void* dBuffer = NULL;
    if(bufferSize > 0)
    {
        cudaMalloc(&dBuffer, bufferSize);
    }

    // Compute the sparse matrix - dense matrix product
    CUSPARSE_CHECK(
        cusparseSpMM(handle, transA, transB, alpha, matA_descr, matB_descr, beta,
                     matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // Clean up
    CUSPARSE_CHECK(cusparseDestroySpMat(matA_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC_descr));

    // if(bufferSize > 0)
    // {
    //     cudaFreeAsync(dBuffer);
    // }
}
#endif

/* Level 3 */
void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
#if (CUDART_VERSION >= 11000)
  generic_SpMM2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc, CUDA_R_32F);
#else
  CUSPARSE_CHECK(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
#endif
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif

#if (CUDART_VERSION >= 11000)
  generic_SpMM2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc, CUDA_R_64F);
#else
  CUSPARSE_CHECK(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  // TODO: adopt these parameters (origin -> dst)
  // &alpha -> &one
  // desc -> descr
  // csrvala -> csr_val.device_data()
  // csrrowptra -> csr_row_ptr.device_data() 
  // csrcolinda -> csr_col_ind.device_data()
  // b -> dense_mat.device_data()
  // i_ldb -> n
  // &beta -> &zero, 
  // c -> result.device_data()
  // i_ldc -> m

  // TODO: I think this leaks the matrix descriptor.  Proper fix is to create
  // real descriptor classes
#endif
}

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  AT_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  CUSPARSE_CHECK(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  // TODO: I think this leaks the matrix descriptor.
}

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}

#endif

}}}} // namespace at::native::sparse::cuda