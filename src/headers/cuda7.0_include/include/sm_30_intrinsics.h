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

#if !defined(__SM_30_INTRINSICS_H__)
#define __SM_30_INTRINSICS_H__

#if defined(__CUDACC_RTC__)
#define __SM_30_INTRINSICS_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_30_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*  Below are declarations of SM-3.0 intrinsics which are included as           *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/

#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

// Warp register exchange (shuffle) intrinsics.
// Notes:
// a) Warp size is hardcoded to 32 here, because the compiler does not know
//    the "warpSize" constant at this time
// b) we cannot map the float __shfl to the int __shfl because it'll mess with
//    the register number (especially if you're doing two shfls to move a double).
__SM_30_INTRINSICS_DECL__ int __shfl(int var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned int __shfl(unsigned int var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ int __shfl_up(int var, unsigned int delta, int width=warpSize); 
__SM_30_INTRINSICS_DECL__ unsigned int __shfl_up(unsigned int var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ int __shfl_down(int var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_down(unsigned int var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ int __shfl_xor(int var, int laneMask, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_xor(unsigned int var, int laneMask, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ float __shfl(float var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ float __shfl_up(float var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ float __shfl_down(float var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ float __shfl_xor(float var, int laneMask, int width=warpSize); 

// 64-bits SHFL
__SM_30_INTRINSICS_DECL__ long long __shfl(long long var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl(unsigned long long var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long long __shfl_up(long long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long long __shfl_down(long long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long long __shfl_xor(long long var, int laneMask, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ double __shfl(double var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ double __shfl_up(double var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ double __shfl_down(double var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ double __shfl_xor(double var, int laneMask, int width=warpSize); 

// long needs some help to choose between 32-bits and 64-bits

__SM_30_INTRINSICS_DECL__ long __shfl(long var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long __shfl(unsigned long var, int srcLane, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long __shfl_up(long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_up(unsigned long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long __shfl_down(long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_down(unsigned long var, unsigned int delta, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ long __shfl_xor(long var, int laneMask, int width=warpSize); 

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_xor(unsigned long var, int laneMask, int width=warpSize); 

#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 300 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_30_INTRINSICS_DECL__

#if !defined(__CUDACC_RTC__)
#include "sm_30_intrinsics.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__SM_30_INTRINSICS_H__ */
