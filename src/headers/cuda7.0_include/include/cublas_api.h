/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 
/*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines) 
 * on top of the CUDA runtime. 
 */

#if !defined(CUBLAS_API_H_)
#define CUBLAS_API_H_

#ifndef CUBLASWINAPI
#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI 
#endif
#endif

#ifndef CUBLASAPI
#error "This file should not be included without defining CUBLASAPI"
#endif

#include "driver_types.h"
#include "cuComplex.h"   /* import complex data type */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* CUBLAS status type returns */
typedef enum{
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
} cublasStatus_t;


typedef enum {
    CUBLAS_FILL_MODE_LOWER=0, 
    CUBLAS_FILL_MODE_UPPER=1
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT=0, 
    CUBLAS_DIAG_UNIT=1
} cublasDiagType_t; 

typedef enum {
    CUBLAS_SIDE_LEFT =0, 
    CUBLAS_SIDE_RIGHT=1
} cublasSideMode_t; 


typedef enum {
    CUBLAS_OP_N=0,  
    CUBLAS_OP_T=1,  
    CUBLAS_OP_C=2  
} cublasOperation_t;


typedef enum { 
    CUBLAS_POINTER_MODE_HOST   = 0,  
    CUBLAS_POINTER_MODE_DEVICE = 1        
} cublasPointerMode_t;

typedef enum { 
    CUBLAS_ATOMICS_NOT_ALLOWED   = 0,  
    CUBLAS_ATOMICS_ALLOWED       = 1        
} cublasAtomicsMode_t;

/* Opaque structure holding CUBLAS library context */
struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2 (cublasHandle_t *handle);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2 (cublasHandle_t handle);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int *version);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2 (cublasHandle_t handle, cudaStream_t streamId); 
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2 (cublasHandle_t handle, cudaStream_t *streamId); 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2 (cublasHandle_t handle, cublasPointerMode_t *mode);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2 (cublasHandle_t handle, cublasPointerMode_t mode);         

CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);
CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);         

/* 
 * cublasStatus_t 
 * cublasSetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy) 
 *
 * copies n elements from a vector x in CPU memory space to a vector y 
 * in GPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, y points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetVector (int n, int elemSize, const void *x, 
                                             int incx, void *devicePtr, int incy);

/* 
 * cublasStatus_t 
 * cublasGetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy)
 * 
 * copies n elements from a vector x in GPU memory space to a vector y 
 * in CPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, x points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetVector (int n, int elemSize, const void *x, 
                                             int incx, void *y, int incy);

/*
 * cublasStatus_t 
 * cublasSetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in CPU memory
 * space to a matrix B in GPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, B points to an object, or part of an 
 * object, that was allocated via cublasAlloc().
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
 *                                ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B, 
                                             int ldb);

/*
 * cublasStatus_t 
 * cublasGetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in GPU memory
 * space to a matrix B in CPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, A points to an object, or part of an 
 * object, that was allocated via cublasAlloc().
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetMatrix (int rows, int cols, int elemSize, 
                                             const void *A, int lda, void *B,
                                             int ldb);

/* 
 * cublasStatus 
 * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx, 
 *                       void *y, int incy, cudaStream_t stream );
 *
 * cublasSetVectorAsync has the same functionnality as cublasSetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetVectorAsync (int n, int elemSize, 
                                                  const void *hostPtr, int incx, 
                                                  void *devicePtr, int incy,
                                                  cudaStream_t stream);
/* 
 * cublasStatus 
 * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx, 
 *                       void *y, int incy, cudaStream_t stream)
 * 
 * cublasGetVectorAsync has the same functionnality as cublasGetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetVectorAsync (int n, int elemSize,
                                                  const void *devicePtr, int incx,
                                                  void *hostPtr, int incy,
                                                  cudaStream_t stream);

/*
 * cublasStatus_t 
 * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
 *                       int lda, void *B, int ldb, cudaStream_t stream)
 *
 * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
 *                                ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream);

/*
 * cublasStatus_t 
 * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A, 
 *                       int lda, void *B, int ldb, cudaStream_t stream)
 *
 * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream);


CUBLASAPI void CUBLASWINAPI cublasXerbla (const char *srName, int info);
/* ---------------- CUBLAS BLAS1 functions ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      float *result);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      double *result);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2 (cublasHandle_t handle,
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     const float *y, 
                                                     int incy,
                                                     float *result);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2 (cublasHandle_t handle,
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     const double *y,
                                                     int incy,
                                                     double *result);  /* host or device pointer */
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      const cuComplex *y, 
                                                      int incy,
                                                      cuComplex *result);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      const cuComplex *y, 
                                                      int incy,
                                                      cuComplex *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      const cuDoubleComplex *y, 
                                                      int incy,
                                                      cuDoubleComplex *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *y, 
                                                      int incy,
                                                      cuDoubleComplex *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *alpha,  /* host or device pointer */
                                                     float *x, 
                                                     int incx);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *alpha,  /* host or device pointer */
                                                     double *x, 
                                                     int incx);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const cuComplex *alpha, /* host or device pointer */
                                                     cuComplex *x, 
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      cuComplex *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const cuDoubleComplex *alpha, /* host or device pointer */
                                                     cuDoubleComplex *x, 
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      cuDoubleComplex *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *alpha, /* host or device pointer */
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *alpha, /* host or device pointer */
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y,
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y,
                                                      int incy);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      float *x, 
                                                      int incx, 
                                                      float *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      double *x, 
                                                      int incx, 
                                                      double *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      cuComplex *x, 
                                                      int incx, 
                                                      cuComplex *y,
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2 (cublasHandle_t handle,
                                                      int n, 
                                                      cuDoubleComplex *x, 
                                                      int incx, 
                                                      cuDoubleComplex *y,
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const float *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const double *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      int *result); /* host or device pointer */
 
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const float *x, 
                                                     int incx, 
                                                     float *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     const double *x, 
                                                     int incx, 
                                                     double *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuComplex *x, 
                                                      int incx, 
                                                      float *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, 
                                                      int n, 
                                                      const cuDoubleComplex *x, 
                                                      int incx, 
                                                      double *result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     cuComplex *x, 
                                                     int incx, 
                                                     cuComplex *y, 
                                                     int incy, 
                                                     const float *c,      /* host or device pointer */
                                                     const cuComplex *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     cuComplex *x, 
                                                     int incx, 
                                                     cuComplex *y, 
                                                     int incy, 
                                                     const float *c,  /* host or device pointer */
                                                     const float *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2 (cublasHandle_t handle, 
                                                     int n, 
                                                     cuDoubleComplex *x, 
                                                     int incx, 
                                                     cuDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,            /* host or device pointer */
                                                     const cuDoubleComplex *s);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     cuDoubleComplex *x, 
                                                     int incx, 
                                                     cuDoubleComplex *y, 
                                                     int incy, 
                                                     const double *c,  /* host or device pointer */
                                                     const double *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, 
                                                     float *a,   /* host or device pointer */
                                                     float *b,   /* host or device pointer */
                                                     float *c,   /* host or device pointer */
                                                     float *s);  /* host or device pointer */
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, 
                                                     double *a,  /* host or device pointer */
                                                     double *b,  /* host or device pointer */
                                                     double *c,  /* host or device pointer */
                                                     double *s); /* host or device pointer */
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, 
                                                     cuComplex *a,  /* host or device pointer */
                                                     cuComplex *b,  /* host or device pointer */
                                                     float *c,      /* host or device pointer */
                                                     cuComplex *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, 
                                                     cuDoubleComplex *a,  /* host or device pointer */
                                                     cuDoubleComplex *b,  /* host or device pointer */
                                                     double *c,           /* host or device pointer */
                                                     cuDoubleComplex *s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     float *x, 
                                                     int incx, 
                                                     float *y, 
                                                     int incy, 
                                                     const float* param);  /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, 
                                                     int n, 
                                                     double *x, 
                                                     int incx, 
                                                     double *y, 
                                                     int incy, 
                                                     const double* param);  /* host or device pointer */
        
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, 
                                                      float *d1,        /* host or device pointer */
                                                      float *d2,        /* host or device pointer */
                                                      float *x1,        /* host or device pointer */
                                                      const float *y1,  /* host or device pointer */
                                                      float *param);    /* host or device pointer */
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, 
                                                      double *d1,        /* host or device pointer */  
                                                      double *d2,        /* host or device pointer */  
                                                      double *x1,        /* host or device pointer */  
                                                      const double *y1,  /* host or device pointer */  
                                                      double *param);    /* host or device pointer */  

/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m, 
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x, 
                                                      int incx, 
                                                      const float *beta,  /* host or device pointer */
                                                      float *y, 
                                                      int incy);  
 
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */
                                                      double *y, 
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x, 
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda, 
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *y,
                                                      int incy);
/* GBMV */                                
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda, 
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */  
                                                      float *y,
                                                      int incy);                                
                                
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda, 
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda, 
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy);                                             
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2 (cublasHandle_t handle,
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku, 
                                                      const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda, 
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy);   
                                         
/* TRMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx);
                                        
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx);
                                                                                                             
/* TBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx);
                                               
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx);
                                         
/* TPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx);                                                 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *AP, 
                                                      cuComplex *x, 
                                                      int incx);
                                                
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *AP, 
                                                      cuDoubleComplex *x, 
                                                      int incx);

/* TRSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);                                                 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx);

/* TPSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const float *AP, 
                                                      float *x, 
                                                      int incx);  
                                                                                                            
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const double *AP, 
                                                      double *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuComplex *AP, 
                                                      cuComplex *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      const cuDoubleComplex *AP, 
                                                      cuDoubleComplex *x, 
                                                      int incx);
/* TBSV */                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx);
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuComplex *A, 
                                                      int lda, 
                                                      cuComplex *x, 
                                                      int incx);
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      cublasOperation_t trans, 
                                                      cublasDiagType_t diag, 
                                                      int n, 
                                                      int k, 
                                                      const cuDoubleComplex *A, 
                                                      int lda, 
                                                      cuDoubleComplex *x, 
                                                      int incx);     
                                         
/* SYMV/HEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const float *alpha, /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta, /* host or device pointer */ 
                                                      float *y,
                                                      int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta, /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy);                                     
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy);                                            
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy);                                     
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */ 
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta,   /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy);   
                                     
/* SBMV/HBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const float *alpha,   /* host or device pointer */ 
                                                      const float *A,
                                                      int lda,
                                                      const float *x, 
                                                      int incx,
                                                      const float *beta,  /* host or device pointer */ 
                                                      float *y,
                                                      int incy);
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const double *alpha,   /* host or device pointer */ 
                                                      const double *A,
                                                      int lda,
                                                      const double *x, 
                                                      int incx,
                                                      const double *beta,   /* host or device pointer */ 
                                                      double *y,
                                                      int incy);
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */ 
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x, 
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */ 
                                                      cuComplex *y,
                                                      int incy);
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, 
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x, 
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */ 
                                                      cuDoubleComplex *y,
                                                      int incy);                                                                            
                                                                                                                                                   
/* SPMV/HPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha,  /* host or device pointer */                                           
                                                      const float *AP,
                                                      const float *x,
                                                      int incx,
                                                      const float *beta,   /* host or device pointer */  
                                                      float *y,
                                                      int incy);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *AP,
                                                      const double *x,
                                                      int incx,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *y,
                                                      int incy);                                     
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2 (cublasHandle_t handle, 
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *AP,
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *y,
                                                      int incy);
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *AP,
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *y, 
                                                      int incy);

/* GER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2 (cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     const float *y,
                                                     int incy,
                                                     float *A,
                                                     int lda);
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2 (cublasHandle_t handle, 
                                                     int m,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */   
                                                     const double *x,
                                                     int incx,
                                                     const double *y,
                                                     int incy,
                                                     double *A,
                                                     int lda);
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2 (cublasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *A,
                                                      int lda);                                   

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2 (cublasHandle_t handle, 
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2 (cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda); 
                                    
/* SYR/HER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *A, 
                                                     int lda);
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *A, 
                                                     int lda);  
                                        
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A, 
                                                     int lda);
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A, 
                                                     int lda);                                          
                                                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *A, 
                                                     int lda); 
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *A, 
                                                     int lda); 

/* SPR/HPR */                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const float *x,
                                                     int incx,
                                                     float *AP);
                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const double *x,
                                                     int incx,
                                                     double *AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha, /* host or device pointer */  
                                                     const cuComplex *x,
                                                     int incx,
                                                     cuComplex *AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2 (cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *x,
                                                     int incx,
                                                     cuDoubleComplex *AP);                       
    
/* SYR2/HER2 */                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *A,
                                                      int lda);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *x,
                                                      int incx,
                                                      const double *y,
                                                      int incy,
                                                      double *A,
                                                      int lda);
                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n, 
                                                      const cuComplex *alpha,  /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx, 
                                                      const cuComplex *y,
                                                      int incy, 
                                                      cuComplex *A, 
                                                      int lda);   
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda);                       
    

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n, 
                                                      const cuComplex *alpha,  /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx, 
                                                      const cuComplex *y,
                                                      int incy, 
                                                      cuComplex *A, 
                                                      int lda);   

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n, 
                                                      const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *A,
                                                      int lda);                       

/* SPR2/HPR2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const float *x,
                                                      int incx,
                                                      const float *y,
                                                      int incy,
                                                      float *AP);
                                                                          
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *x,
                                                      int incx, 
                                                      const double *y,
                                                      int incy,
                                                      double *AP);
    

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *x,
                                                      int incx,
                                                      const cuComplex *y,
                                                      int incy,
                                                      cuComplex *AP);
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *x,
                                                      int incx,
                                                      const cuDoubleComplex *y,
                                                      int incy,
                                                      cuDoubleComplex *AP); 

/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A, 
                                                      int lda,
                                                      const float *B,
                                                      int ldb, 
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda,
                                                      const double *B,
                                                      int ldb, 
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc);
                                        
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A, 
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb, 
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc);
                                        
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A, 
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb, 
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc);                                                                                
                            
/* SYRK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      double *C,
                                                      int ldc);   
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc);         
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C, 
                                                      int ldc);
/* HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,  /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const float *beta,   /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2 (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,  /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const double *beta,  /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc);    

/* SYR2K */                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha, /* host or device pointer */  
                                                       const float *A,
                                                       int lda,
                                                       const float *B,
                                                       int ldb,
                                                       const float *beta, /* host or device pointer */  
                                                       float *C,
                                                       int ldc);  
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const double *alpha, /* host or device pointer */  
                                                       const double *A,
                                                       int lda,
                                                       const double *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       double *C,
                                                       int ldc);
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */  
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const cuComplex *beta, /* host or device pointer */  
                                                       cuComplex *C,
                                                       int ldc);
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha,  /* host or device pointer */  
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const cuDoubleComplex *beta,  /* host or device pointer */  
                                                       cuDoubleComplex *C,
                                                       int ldc);  
/* HER2K */                                       
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha, /* host or device pointer */  
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const float *beta,   /* host or device pointer */  
                                                       cuComplex *C,
                                                       int ldc);  
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2 (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t trans, 
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha, /* host or device pointer */  
                                                       const cuDoubleComplex *A, 
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const double *beta, /* host or device pointer */  
                                                       cuDoubleComplex *C,
                                                       int ldc);     
/* SYRKX : eXtended SYRK*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float *alpha, /* host or device pointer */ 
                                                    const float *A,
                                                    int lda,
                                                    const float *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    float *C,
                                                    int ldc);
                                                   
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const double *alpha, /* host or device pointer */ 
                                                    const double *A,
                                                    int lda,
                                                    const double *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    double *C,
                                                    int ldc);
                                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */ 
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const cuComplex *beta, /* host or device pointer */ 
                                                    cuComplex *C, 
                                                    int ldc);
                                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo, 
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const cuDoubleComplex *beta, /* host or device pointer */ 
                                                    cuDoubleComplex *C, 
                                                    int ldc);
/* HERKX : eXtended HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha, /* host or device pointer */ 
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const float *beta, /* host or device pointer */ 
                                                    cuComplex *C,
                                                    int ldc);
                                                
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const double *beta, /* host or device pointer */ 
                                                    cuDoubleComplex *C,
                                                    int ldc);
/* SYMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m, 
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc);                                     

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc);
                                                   
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc);   
                                     
/* HEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C, 
                                                      int ldc); 

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc); 
    
/* TRSM */                                                                         
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2 (cublasHandle_t handle, 
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda,
                                                      float *B,
                                                      int ldb);
    

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda, 
                                                      double *B,
                                                      int ldb);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *B,
                                                     int ldb);
                  
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, 
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *A,                                        
                                                     int lda,
                                                     cuDoubleComplex *B,
                                                     int ldb);              
                                                
 /* TRMM */  
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha, /* host or device pointer */  
                                                      const float *A,
                                                      int lda, 
                                                      const float *B,
                                                      int ldb,
                                                      float *C,
                                                      int ldc);
                                               
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2 (cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      double *C,
                                                      int ldc);
                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha, /* host or device pointer */  
                                                     const cuComplex *A,
                                                     int lda,
                                                     const cuComplex *B,
                                                     int ldb,
                                                     cuComplex *C,
                                                     int ldc);
                  
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, 
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha, /* host or device pointer */  
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     const cuDoubleComplex *B,
                                                     int ldb,
                                                     cuDoubleComplex *C,
                                                     int ldc);
/* BATCH GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const float *alpha,  /* host or device pointer */  
                                                          const float *Aarray[], 
                                                          int lda,
                                                          const float *Barray[],
                                                          int ldb, 
                                                          const float *beta,   /* host or device pointer */  
                                                          float *Carray[],
                                                          int ldc,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const double *alpha,  /* host or device pointer */ 
                                                          const double *Aarray[], 
                                                          int lda,
                                                          const double *Barray[],
                                                          int ldb, 
                                                          const double *beta,  /* host or device pointer */ 
                                                          double *Carray[],
                                                          int ldc,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha, /* host or device pointer */ 
                                                          const cuComplex *Aarray[], 
                                                          int lda,
                                                          const cuComplex *Barray[],
                                                          int ldb, 
                                                          const cuComplex *beta, /* host or device pointer */ 
                                                          cuComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                          const cuDoubleComplex *Aarray[], 
                                                          int lda,
                                                          const cuDoubleComplex *Barray[],
                                                          int ldb, 
                                                          const cuDoubleComplex *beta, /* host or device pointer */ 
                                                          cuDoubleComplex *Carray[],
                                                          int ldc,
                                                          int batchCount); 

/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const float *alpha, /* host or device pointer */ 
                                                  const float *A, 
                                                  int lda,
                                                  const float *beta , /* host or device pointer */ 
                                                  const float *B, 
                                                  int ldb,
                                                  float *C, 
                                                  int ldc);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const double *alpha, /* host or device pointer */ 
                                                  const double *A, 
                                                  int lda,
                                                  const double *beta, /* host or device pointer */ 
                                                  const double *B, 
                                                  int ldb,
                                                  double *C, 
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const cuComplex *alpha, /* host or device pointer */ 
                                                  const cuComplex *A, 
                                                  int lda,
                                                  const cuComplex *beta, /* host or device pointer */  
                                                  const cuComplex *B, 
                                                  int ldb,
                                                  cuComplex *C, 
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa, 
                                                  cublasOperation_t transb,
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *alpha, /* host or device pointer */ 
                                                  const cuDoubleComplex *A, 
                                                  int lda,
                                                  const cuDoubleComplex *beta, /* host or device pointer */  
                                                  const cuDoubleComplex *B, 
                                                  int ldb,
                                                  cuDoubleComplex *C, 
                                                  int ldc);
 
/* Batched LU - GETRF*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  float *A[],                      /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  double *A[],                     /*Device pointer*/
                                                  int lda, 
                                                  int *P,                          /*Device Pointer*/
                                                  int *info,                       /*Device Pointer*/
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuComplex *A[],                 /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                  int n, 
                                                  cuDoubleComplex *A[],           /*Device pointer*/
                                                  int lda, 
                                                  int *P,                         /*Device Pointer*/
                                                  int *info,                      /*Device Pointer*/
                                                  int batchSize);

/* Batched inversion based on LU factorization from getrf */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const float *A[],               /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  float *C[],                     /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const double *A[],              /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  double *C[],                    /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuComplex *A[],            /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuComplex *C[],                 /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuDoubleComplex *A[],     /*Device pointer*/
                                                  int lda,
                                                  const int *P,                   /*Device pointer*/
                                                  cuDoubleComplex *C[],           /*Device pointer*/
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

/* Batched solver based on LU factorization from getrf */

CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasSgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const float *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            float *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int n, 
                                                           int nrhs, 
                                                           const double *Aarray[], 
                                                           int lda, 
                                                           const int *devIpiv, 
                                                           double *Barray[], 
                                                           int ldb, 
                                                           int *info,
                                                           int batchSize);

CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasCgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);


CUBLASAPI cublasStatus_t  CUBLASWINAPI cublasZgetrsBatched( cublasHandle_t handle, 
                                                            cublasOperation_t trans, 
                                                            int n, 
                                                            int nrhs, 
                                                            const cuDoubleComplex *Aarray[], 
                                                            int lda, 
                                                            const int *devIpiv, 
                                                            cuDoubleComplex *Barray[], 
                                                            int ldb, 
                                                            int *info,
                                                            int batchSize);



/* TRSM - Batched Triangular Solver */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const float *alpha,           /*Host or Device Pointer*/
                                                          const float *A[], 
                                                          int lda,
                                                          float *B[], 
                                                          int ldb,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const double *alpha,          /*Host or Device Pointer*/
                                                          const double *A[], 
                                                          int lda,
                                                          double *B[], 
                                                          int ldb,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const cuComplex *alpha,       /*Host or Device Pointer*/
                                                          const cuComplex *A[], 
                                                          int lda,
                                                          cuComplex *B[], 
                                                          int ldb,
                                                          int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched( cublasHandle_t    handle, 
                                                          cublasSideMode_t  side, 
                                                          cublasFillMode_t  uplo,
                                                          cublasOperation_t trans, 
                                                          cublasDiagType_t  diag,
                                                          int m, 
                                                          int n, 
                                                          const cuDoubleComplex *alpha, /*Host or Device Pointer*/
                                                          const cuDoubleComplex *A[], 
                                                          int lda,
                                                          cuDoubleComplex *B[], 
                                                          int ldb,
                                                          int batchCount);

/* Batched - MATINV*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const float *A[],                  /*Device pointer*/
                                                          int lda, 
                                                          float *Ainv[],               /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const double *A[],                 /*Device pointer*/
                                                          int lda, 
                                                          double *Ainv[],              /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuComplex *A[],              /*Device pointer*/
                                                          int lda, 
                                                          cuComplex *Ainv[],           /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                          int n, 
                                                          const cuDoubleComplex *A[],        /*Device pointer*/
                                                          int lda, 
                                                          cuDoubleComplex *Ainv[],     /*Device pointer*/
                                                          int lda_inv, 
                                                          int *info,                   /*Device Pointer*/
                                                          int batchSize);

/* Batch QR Factorization */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched( cublasHandle_t handle, 
                                                           int m, 
                                                           int n,
                                                           float *Aarray[],           /*Device pointer*/
                                                           int lda, 
                                                           float *TauArray[],        /* Device pointer*/                                                           
                                                           int *info,
                                                           int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasDgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            double *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            double *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasCgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            cuComplex *TauArray[],        /* Device pointer*/                                                            
                                                            int *info,
                                                            int batchSize);
                                                            
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasZgeqrfBatched( cublasHandle_t handle, 
                                                            int m, 
                                                            int n,
                                                            cuDoubleComplex *Aarray[],           /*Device pointer*/
                                                            int lda, 
                                                            cuDoubleComplex *TauArray[],        /* Device pointer*/                                                          
                                                            int *info,
                                                            int batchSize);
/* Least Square Min only m >= n and Non-transpose supported */
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasSgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           float *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           float *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize );
                                                                
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasDgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t trans,  
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           double *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           double *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray, /* Device pointer*/
                                                           int batchSize);
                                                                
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasCgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           cuComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           cuComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize);
                                                                
CUBLASAPI cublasStatus_t CUBLASWINAPI  cublasZgelsBatched( cublasHandle_t handle, 
                                                           cublasOperation_t trans, 
                                                           int m,  
                                                           int n,
                                                           int nrhs,
                                                           cuDoubleComplex *Aarray[], /*Device pointer*/
                                                           int lda, 
                                                           cuDoubleComplex *Carray[], /* Device pointer*/
                                                           int ldc,                                                                 
                                                           int *info, 
                                                           int *devInfoArray,
                                                           int batchSize);                                                                                                                                                                                                
/* DGMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const float *A, 
                                                  int lda,
                                                  const float *x, 
                                                  int incx,
                                                  float *C, 
                                                  int ldc);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const double *A, 
                                                  int lda,
                                                  const double *x, 
                                                  int incx,
                                                  double *C, 
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const cuComplex *A, 
                                                  int lda,
                                                  const cuComplex *x, 
                                                  int incx,
                                                  cuComplex *C, 
                                                  int ldc);
    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode, 
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *A, 
                                                  int lda,
                                                  const cuDoubleComplex *x, 
                                                  int incx,
                                                  cuDoubleComplex *C, 
                                                  int ldc);

/* TPTTR : Triangular Pack format to Triangular format */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *AP,
                                                     float *A,  
                                                     int lda );
                                       
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *AP,
                                                     double *A,  
                                                     int lda );
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuComplex *AP,
                                                     cuComplex *A,  
                                                     int lda );
                                                    
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpttr ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuDoubleComplex *AP,
                                                     cuDoubleComplex *A,  
                                                     int lda );
 /* TRTTP : Triangular format to Triangular Pack format */                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const float *A,
                                                     int lda,
                                                     float *AP );
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const double *A,
                                                     int lda,
                                                     double *AP );
                                      
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *AP );
                                                     
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrttp ( cublasHandle_t handle, 
                                                     cublasFillMode_t uplo, 
                                                     int n,                                     
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *AP );                                        
                                      
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(CUBLAS_API_H_) */
