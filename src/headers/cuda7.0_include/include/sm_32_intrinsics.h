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

#if !defined(__SM_32_INTRINSICS_H__)
#define __SM_32_INTRINSICS_H__

#if defined(__CUDACC_RTC__)
#define __SM_32_INTRINSICS_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_32_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320

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
*  Below are declarations of SM-3.5 intrinsics which are included as           *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/

__SM_32_INTRINSICS_DECL__ long __ldg(const long *ptr); 
__SM_32_INTRINSICS_DECL__ unsigned long __ldg(const unsigned long *ptr); 


__SM_32_INTRINSICS_DECL__ char __ldg(const char *ptr); 
__SM_32_INTRINSICS_DECL__ short __ldg(const short *ptr); 
__SM_32_INTRINSICS_DECL__ int __ldg(const int *ptr); 
__SM_32_INTRINSICS_DECL__ long long __ldg(const long long *ptr); 
__SM_32_INTRINSICS_DECL__ char2 __ldg(const char2 *ptr); 
__SM_32_INTRINSICS_DECL__ char4 __ldg(const char4 *ptr); 
__SM_32_INTRINSICS_DECL__ short2 __ldg(const short2 *ptr); 
__SM_32_INTRINSICS_DECL__ short4 __ldg(const short4 *ptr); 
__SM_32_INTRINSICS_DECL__ int2 __ldg(const int2 *ptr); 
__SM_32_INTRINSICS_DECL__ int4 __ldg(const int4 *ptr); 
__SM_32_INTRINSICS_DECL__ longlong2 __ldg(const longlong2 *ptr); 

__SM_32_INTRINSICS_DECL__ unsigned char __ldg(const unsigned char *ptr); 
__SM_32_INTRINSICS_DECL__ unsigned short __ldg(const unsigned short *ptr); 
__SM_32_INTRINSICS_DECL__ unsigned int __ldg(const unsigned int *ptr); 
__SM_32_INTRINSICS_DECL__ unsigned long long __ldg(const unsigned long long *ptr); 
__SM_32_INTRINSICS_DECL__ uchar2 __ldg(const uchar2 *ptr); 
__SM_32_INTRINSICS_DECL__ uchar4 __ldg(const uchar4 *ptr); 
__SM_32_INTRINSICS_DECL__ ushort2 __ldg(const ushort2 *ptr); 
__SM_32_INTRINSICS_DECL__ ushort4 __ldg(const ushort4 *ptr); 
__SM_32_INTRINSICS_DECL__ uint2 __ldg(const uint2 *ptr); 
__SM_32_INTRINSICS_DECL__ uint4 __ldg(const uint4 *ptr); 
__SM_32_INTRINSICS_DECL__ ulonglong2 __ldg(const ulonglong2 *ptr); 

__SM_32_INTRINSICS_DECL__ float __ldg(const float *ptr); 
__SM_32_INTRINSICS_DECL__ double __ldg(const double *ptr); 
__SM_32_INTRINSICS_DECL__ float2 __ldg(const float2 *ptr); 
__SM_32_INTRINSICS_DECL__ float4 __ldg(const float4 *ptr); 
__SM_32_INTRINSICS_DECL__ double2 __ldg(const double2 *ptr); 


// SHF is the "funnel shift" operation - an accelerated left/right shift with carry
// operating on 64-bit quantities, which are concatenations of two 32-bit registers.

// This shifts [b:a] left by "shift" bits, returning the most significant bits of the result.
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift);
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift);

// This shifts [b:a] right by "shift" bits, returning the least significant bits of the result.
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift);
__SM_32_INTRINSICS_DECL__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift);


#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 320 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_32_INTRINSICS_DECL__

#if !defined(__CUDACC_RTC__)
#include "sm_32_intrinsics.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__SM_32_INTRINSICS_H__ */
