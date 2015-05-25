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

#if !defined(__DEVICE_ATOMIC_FUNCTIONS_H__)
#define __DEVICE_ATOMIC_FUNCTIONS_H__

#if defined(__CUDACC_RTC__)
#define __DEVICE_ATOMIC_FUNCTIONS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __DEVICE_ATOMIC_FUNCTIONS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 110

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "host_defines.h"

#if !defined(__CUDACC_RTC__)
extern "C"
{
#endif /* !__CUDACC_RTC__ */
extern __device__ __device_builtin__ int          __iAtomicAdd(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicAdd(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicExch(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicExch(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ float        __fAtomicExch(float *address, float val);
extern __device__ __device_builtin__ int          __iAtomicMin(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicMin(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicMax(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicMax(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ unsigned int __uAtomicDec(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicAnd(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicAnd(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicOr(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicOr(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicXor(int *address, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicXor(unsigned int *address, unsigned int val);
extern __device__ __device_builtin__ int          __iAtomicCAS(int *address, int compare, int val);
extern __device__ __device_builtin__ unsigned int __uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val);
#if !defined(__CUDACC_RTC__)
}
#endif /* !__CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicAdd(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAdd(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicSub(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicSub(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicExch(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicExch(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ float atomicExch(float *address, float val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicMin(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMin(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicMax(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMax(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicInc(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicDec(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicAnd(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAnd(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicOr(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicOr(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicXor(int *address, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicXor(unsigned int *address, unsigned int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicCAS(int *address, int compare, int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val);

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 110 */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 120

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "host_defines.h"

#if !defined(__CUDACC_RTC__)
extern "C"
{
#endif /* !__CUDACC_RTC__ */
extern __device__ __device_builtin__ unsigned long long int __ullAtomicAdd(unsigned long long int *address, unsigned long long int val);
extern __device__ __device_builtin__ unsigned long long int __ullAtomicExch(unsigned long long int *address, unsigned long long int val);
extern __device__ __device_builtin__ unsigned long long int __ullAtomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);
extern __device__ __device_builtin__ int                    __any(int cond);
extern __device__ __device_builtin__ int                    __all(int cond);
#if !defined(__CUDACC_RTC__)
}
#endif /* !__CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ bool any(bool cond);

__DEVICE_ATOMIC_FUNCTIONS_DECL__ bool all(bool cond);

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 120 */

#endif /* __cplusplus && __CUDACC__ */

#undef __DEVICE_ATOMIC_FUNCTIONS_DECL__

#if !defined(__CUDACC_RTC__)
#include "device_atomic_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__DEVICE_ATOMIC_FUNCTIONS_H__ */
