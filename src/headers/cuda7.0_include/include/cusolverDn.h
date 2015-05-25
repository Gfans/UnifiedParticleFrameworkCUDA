/*
 * Copyright 2014 NVIDIA Corporation.  All rights reserved.
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
 
 /*   cuSolverDN : Dense Linear Algebra Library

 */
 
#if !defined(CUSOLVERDN_H_)
#define CUSOLVERDN_H_

#ifndef CUDENSEAPI
#ifdef _WIN32
#define CUDENSEAPI __stdcall
#else
#define CUDENSEAPI 
#endif
#endif

#include "driver_types.h"
#include "cuComplex.h"   /* import complex data type */
#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#ifndef CUSOLVER_COMMON
#define CUSOLVER_COMMON
#include "cusolver_common.h"
#endif // CUSOLVER_COMMON

struct cusolverDnContext;
typedef struct cusolverDnContext *cusolverDnHandle_t;

cusolverStatus_t CUDENSEAPI cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t CUDENSEAPI cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t CUDENSEAPI cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t CUDENSEAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);

/* Cholesky factorization and its solver */
cusolverStatus_t CUDENSEAPI cusolverDnSpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *Lwork);

cusolverStatus_t CUDENSEAPI cusolverDnSpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda,  
    float *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


cusolverStatus_t CUDENSEAPI cusolverDnSpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const float *A,
    int lda,
    float *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnDpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const double *A,
    int lda,
    double *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnCpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnZpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    int *devInfo);


/* LU Factorization */
cusolverStatus_t CUDENSEAPI cusolverDnSgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *Lwork );


cusolverStatus_t CUDENSEAPI cusolverDnSgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

/* Row pivoting */
cusolverStatus_t CUDENSEAPI cusolverDnSlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    float *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUDENSEAPI cusolverDnDlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    double *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUDENSEAPI cusolverDnClaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUDENSEAPI cusolverDnZlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

/* LU solve */
cusolverStatus_t CUDENSEAPI cusolverDnSgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const float *A, 
    int lda, 
    const int *devIpiv, 
    float *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const double *A, 
    int lda, 
    const int *devIpiv, 
    double *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuComplex *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuDoubleComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuDoubleComplex *B, 
    int ldb, 
    int *devInfo );

/* QR factorization */
cusolverStatus_t CUDENSEAPI cusolverDnSgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda, 
    float *TAU,  
    float *Workspace,  
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *TAU, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *TAU, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *TAU, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnSormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnDormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnCunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUDENSEAPI cusolverDnZunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);


/* QR factorization workspace query */
cusolverStatus_t CUDENSEAPI cusolverDnSgeqrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDgeqrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCgeqrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZgeqrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *Lwork );


/* bidiagonal */
cusolverStatus_t CUDENSEAPI cusolverDnSgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda,
    float *D, 
    float *E, 
    float *TAUQ,  
    float *TAUP, 
    float *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *D, 
    double *E, 
    double *TAUQ, 
    double *TAUP, 
    double *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    float *D, 
    float *E, 
    cuComplex *TAUQ, 
    cuComplex *TAUP,
    cuComplex *Work, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A,
    int lda, 
    double *D, 
    double *E, 
    cuDoubleComplex *TAUQ,
    cuDoubleComplex *TAUP, 
    cuDoubleComplex *Work, 
    int Lwork, 
    int *devInfo );


cusolverStatus_t CUDENSEAPI cusolverDnSsytrd (cusolverDnHandle_t handle, char uplo, int n, float *A, int
        lda, float *D, float *E, float *tau, float *Work, int Lwork, int *info);
cusolverStatus_t CUDENSEAPI cusolverDnDsytrd (cusolverDnHandle_t handle, char uplo, int n, double *A, int
        lda, double *D, double *E, double *tau, double *Work, int Lwork, int
        *info);
/* bidiagonal factorization workspace query */
cusolverStatus_t CUDENSEAPI cusolverDnSgebrd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDgebrd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCgebrd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZgebrd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

/* singular value decomposition, A = U * Sigma * V^H */
cusolverStatus_t CUDENSEAPI cusolverDnSgesvd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDgesvd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCgesvd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZgesvd_bufferSize( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnSgesvd (
    cusolverDnHandle_t handle, 
    char jobu, 
    char jobvt, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U, 
    int ldu, 
    float *VT, 
    int ldvt, 
    float *Work, 
    int Lwork, 
    float *rwork, 
    int  *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDgesvd (
    cusolverDnHandle_t handle, 
    char jobu, 
    char jobvt, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *S, 
    double *U, 
    int ldu, 
    double *VT, 
    int ldvt, 
    double *Work,
    int Lwork, 
    double *rwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCgesvd (
    cusolverDnHandle_t handle, 
    char jobu, 
    char jobvt, 
    int m, 
    int n, 
    cuComplex *A,
    int lda, 
    float *S, 
    cuComplex *U, 
    int ldu, 
    cuComplex *VT, 
    int ldvt,
    cuComplex *Work, 
    int Lwork, 
    float *rwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZgesvd (
    cusolverDnHandle_t handle, 
    char jobu, 
    char jobvt, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    double *S, 
    cuDoubleComplex *U, 
    int ldu, 
    cuDoubleComplex *VT, 
    int ldvt, 
    cuDoubleComplex *Work, 
    int Lwork, 
    double *rwork, 
    int *devInfo );

/* LDLT,UDUT factorization */
cusolverStatus_t CUDENSEAPI cusolverDnSsytrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda, 
    int *ipiv, 
    float *work, 
    int lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnDsytrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    int *ipiv, 
    double *work, 
    int lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnCsytrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *ipiv, 
    cuComplex *work, 
    int lwork, 
    int *devInfo );

cusolverStatus_t CUDENSEAPI cusolverDnZsytrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *ipiv, 
    cuDoubleComplex *work, 
    int lwork, 
    int *devInfo );

/* SYTRF factorization workspace query */
cusolverStatus_t CUDENSEAPI cusolverDnSsytrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnDsytrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnCsytrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUDENSEAPI cusolverDnZsytrf_bufferSize( 
    cusolverDnHandle_t handle, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *Lwork );

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(CUDENSE_H_) */
