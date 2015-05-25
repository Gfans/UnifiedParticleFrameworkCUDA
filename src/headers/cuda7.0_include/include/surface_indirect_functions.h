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


#ifndef __SURFACE_INDIRECT_FUNCTIONS_H__
#define __SURFACE_INDIRECT_FUNCTIONS_H__

#if defined(__CUDACC_RTC__)
#define __SURFACE_INDIRECT_FUNCTIONS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SURFACE_INDIRECT_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200


#include "builtin_types.h"
#include "host_defines.h"
#include "vector_functions.h"


/*******************************************************************************
*                                                                              *
* 1D Surface indirect read functions
*                                                                              *
*******************************************************************************/

extern "C" {
__device__ void __suld_1d_i8_clamp(char *, unsigned long long, int);
__device__ void __suld_1d_i16_clamp(short *, unsigned long long, int);
__device__ void __suld_1d_i32_clamp(int *, unsigned long long, int);
__device__ void __suld_1d_i64_clamp(long long *, unsigned long long, int);
__device__ void __suld_1d_v2i8_clamp(char *, char *, unsigned long long, int);
__device__ void __suld_1d_v2i16_clamp(short *, short *, unsigned long long, int);
__device__ void __suld_1d_v2i32_clamp(int *, int *, unsigned long long, int);
__device__ void __suld_1d_v2i64_clamp(long long *, long long *, unsigned long long, int);
__device__ void __suld_1d_v4i8_clamp(char *, char *, char *, char *, unsigned long long, int);
__device__ void __suld_1d_v4i16_clamp(short *, short *, short *, short *, unsigned long long, int);
__device__ void __suld_1d_v4i32_clamp(int *, int *, int *, int *, unsigned long long, int);
__device__ void __suld_1d_array_i8_clamp(char *, unsigned long long, int, int);
__device__ void __suld_1d_array_i16_clamp(short *, unsigned long long, int, int);
__device__ void __suld_1d_array_i32_clamp(int *, unsigned long long, int, int);
__device__ void __suld_1d_array_i64_clamp(long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i8_clamp(char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i16_clamp(short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i32_clamp(int *, int *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i64_clamp(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i8_clamp(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i16_clamp(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i32_clamp(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_i8_clamp(char *, unsigned long long, int, int);
__device__ void __suld_2d_i16_clamp(short *, unsigned long long, int, int);
__device__ void __suld_2d_i32_clamp(int *, unsigned long long, int, int);
__device__ void __suld_2d_i64_clamp(long long *, unsigned long long, int, int);
__device__ void __suld_2d_v2i8_clamp(char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v2i16_clamp(short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v2i32_clamp(int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_v2i64_clamp(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_2d_v4i8_clamp(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v4i16_clamp(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v4i32_clamp(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_array_i8_clamp(char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i16_clamp(short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i32_clamp(int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i64_clamp(long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i8_clamp(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i16_clamp(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i32_clamp(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i64_clamp(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i8_clamp(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i16_clamp(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i32_clamp(int *, int *, int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i8_clamp(char *, unsigned long long, int, int, int);
__device__ void __suld_3d_i16_clamp(short *, unsigned long long, int, int, int);
__device__ void __suld_3d_i32_clamp(int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i64_clamp(long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i8_clamp(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i16_clamp(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i32_clamp(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i64_clamp(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i8_clamp(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i16_clamp(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i32_clamp(int *, int *, int *, int *, unsigned long long, int, int, int);
__device__ void __suld_1d_i8_trap(char *, unsigned long long, int);
__device__ void __suld_1d_i16_trap(short *, unsigned long long, int);
__device__ void __suld_1d_i32_trap(int *, unsigned long long, int);
__device__ void __suld_1d_i64_trap(long long *, unsigned long long, int);
__device__ void __suld_1d_v2i8_trap(char *, char *, unsigned long long, int);
__device__ void __suld_1d_v2i16_trap(short *, short *, unsigned long long, int);
__device__ void __suld_1d_v2i32_trap(int *, int *, unsigned long long, int);
__device__ void __suld_1d_v2i64_trap(long long *, long long *, unsigned long long, int);
__device__ void __suld_1d_v4i8_trap(char *, char *, char *, char *, unsigned long long, int);
__device__ void __suld_1d_v4i16_trap(short *, short *, short *, short *, unsigned long long, int);
__device__ void __suld_1d_v4i32_trap(int *, int *, int *, int *, unsigned long long, int);
__device__ void __suld_1d_array_i8_trap(char *, unsigned long long, int, int);
__device__ void __suld_1d_array_i16_trap(short *, unsigned long long, int, int);
__device__ void __suld_1d_array_i32_trap(int *, unsigned long long, int, int);
__device__ void __suld_1d_array_i64_trap(long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i8_trap(char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i16_trap(short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i32_trap(int *, int *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i64_trap(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i8_trap(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i16_trap(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i32_trap(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_i8_trap(char *, unsigned long long, int, int);
__device__ void __suld_2d_i16_trap(short *, unsigned long long, int, int);
__device__ void __suld_2d_i32_trap(int *, unsigned long long, int, int);
__device__ void __suld_2d_i64_trap(long long *, unsigned long long, int, int);
__device__ void __suld_2d_v2i8_trap(char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v2i16_trap(short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v2i32_trap(int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_v2i64_trap(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_2d_v4i8_trap(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v4i16_trap(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v4i32_trap(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_array_i8_trap(char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i16_trap(short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i32_trap(int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i64_trap(long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i8_trap(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i16_trap(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i32_trap(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i64_trap(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i8_trap(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i16_trap(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i32_trap(int *, int *, int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i8_trap(char *, unsigned long long, int, int, int);
__device__ void __suld_3d_i16_trap(short *, unsigned long long, int, int, int);
__device__ void __suld_3d_i32_trap(int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i64_trap(long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i8_trap(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i16_trap(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i32_trap(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i64_trap(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i8_trap(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i16_trap(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i32_trap(int *, int *, int *, int *, unsigned long long, int, int, int);
__device__ void __suld_1d_i8_zero(char *, unsigned long long, int);
__device__ void __suld_1d_i16_zero(short *, unsigned long long, int);
__device__ void __suld_1d_i32_zero(int *, unsigned long long, int);
__device__ void __suld_1d_i64_zero(long long *, unsigned long long, int);
__device__ void __suld_1d_v2i8_zero(char *, char *, unsigned long long, int);
__device__ void __suld_1d_v2i16_zero(short *, short *, unsigned long long, int);
__device__ void __suld_1d_v2i32_zero(int *, int *, unsigned long long, int);
__device__ void __suld_1d_v2i64_zero(long long *, long long *, unsigned long long, int);
__device__ void __suld_1d_v4i8_zero(char *, char *, char *, char *, unsigned long long, int);
__device__ void __suld_1d_v4i16_zero(short *, short *, short *, short *, unsigned long long, int);
__device__ void __suld_1d_v4i32_zero(int *, int *, int *, int *, unsigned long long, int);
__device__ void __suld_1d_array_i8_zero(char *, unsigned long long, int, int);
__device__ void __suld_1d_array_i16_zero(short *, unsigned long long, int, int);
__device__ void __suld_1d_array_i32_zero(int *, unsigned long long, int, int);
__device__ void __suld_1d_array_i64_zero(long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i8_zero(char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i16_zero(short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i32_zero(int *, int *, unsigned long long, int, int);
__device__ void __suld_1d_array_v2i64_zero(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i8_zero(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i16_zero(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_1d_array_v4i32_zero(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_i8_zero(char *, unsigned long long, int, int);
__device__ void __suld_2d_i16_zero(short *, unsigned long long, int, int);
__device__ void __suld_2d_i32_zero(int *, unsigned long long, int, int);
__device__ void __suld_2d_i64_zero(long long *, unsigned long long, int, int);
__device__ void __suld_2d_v2i8_zero(char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v2i16_zero(short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v2i32_zero(int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_v2i64_zero(long long *, long long *, unsigned long long, int, int);
__device__ void __suld_2d_v4i8_zero(char *, char *, char *, char *, unsigned long long, int, int);
__device__ void __suld_2d_v4i16_zero(short *, short *, short *, short *, unsigned long long, int, int);
__device__ void __suld_2d_v4i32_zero(int *, int *, int *, int *, unsigned long long, int, int);
__device__ void __suld_2d_array_i8_zero(char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i16_zero(short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i32_zero(int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_i64_zero(long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i8_zero(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i16_zero(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i32_zero(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v2i64_zero(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i8_zero(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i16_zero(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_2d_array_v4i32_zero(int *, int *, int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i8_zero(char *, unsigned long long, int, int, int);
__device__ void __suld_3d_i16_zero(short *, unsigned long long, int, int, int);
__device__ void __suld_3d_i32_zero(int *, unsigned long long, int, int, int);
__device__ void __suld_3d_i64_zero(long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i8_zero(char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i16_zero(short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i32_zero(int *, int *, unsigned long long, int, int, int);
__device__ void __suld_3d_v2i64_zero(long long *, long long *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i8_zero(char *, char *, char *, char *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i16_zero(short *, short *, short *, short *, unsigned long long, int, int, int);
__device__ void __suld_3d_v4i32_zero(int *, int *, int *, int *, unsigned long long, int, int, int);

__device__ void __sust_b_1d_i8_clamp(cudaSurfaceObject_t, int, unsigned char);
__device__ void __sust_b_1d_i16_clamp(cudaSurfaceObject_t, int, unsigned short);
__device__ void __sust_b_1d_i32_clamp(cudaSurfaceObject_t, int, unsigned int);
__device__ void __sust_b_1d_i64_clamp(cudaSurfaceObject_t, int, unsigned long long);
__device__ void __sust_b_1d_v2i8_clamp(cudaSurfaceObject_t, int, uchar2);
__device__ void __sust_b_1d_v2i16_clamp(cudaSurfaceObject_t, int, ushort2);
__device__ void __sust_b_1d_v2i32_clamp(cudaSurfaceObject_t, int, uint2);
__device__ void __sust_b_1d_v2i64_clamp(cudaSurfaceObject_t, int, ulonglong2);
__device__ void __sust_b_1d_v4i8_clamp(cudaSurfaceObject_t, int, uchar4);
__device__ void __sust_b_1d_v4i16_clamp(cudaSurfaceObject_t, int, ushort4);
__device__ void __sust_b_1d_v4i32_clamp(cudaSurfaceObject_t, int, uint4);
__device__ void __sust_b_1d_array_i8_clamp(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_1d_array_i16_clamp(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_1d_array_i32_clamp(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_1d_array_i64_clamp(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_1d_array_v2i8_clamp(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_1d_array_v2i16_clamp(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_1d_array_v2i32_clamp(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_1d_array_v2i64_clamp(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_1d_array_v4i8_clamp(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_1d_array_v4i16_clamp(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_1d_array_v4i32_clamp(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_i8_clamp(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_2d_i16_clamp(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_2d_i32_clamp(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_2d_i64_clamp(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_2d_v2i8_clamp(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_2d_v2i16_clamp(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_2d_v2i32_clamp(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_2d_v2i64_clamp(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_2d_v4i8_clamp(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_2d_v4i16_clamp(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_2d_v4i32_clamp(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_array_i8_clamp(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_2d_array_i16_clamp(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_2d_array_i32_clamp(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_2d_array_i64_clamp(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_2d_array_v2i8_clamp(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_2d_array_v2i16_clamp(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_2d_array_v2i32_clamp(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_2d_array_v2i64_clamp(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_2d_array_v4i8_clamp(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_2d_array_v4i16_clamp(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_2d_array_v4i32_clamp(cudaSurfaceObject_t, int, int, int, uint4);
__device__ void __sust_b_3d_i8_clamp(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_3d_i16_clamp(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_3d_i32_clamp(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_3d_i64_clamp(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_3d_v2i8_clamp(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_3d_v2i16_clamp(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_3d_v2i32_clamp(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_3d_v2i64_clamp(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_3d_v4i8_clamp(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_3d_v4i16_clamp(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_3d_v4i32_clamp(cudaSurfaceObject_t, int, int, int, uint4);
__device__ void __sust_b_1d_i8_trap(cudaSurfaceObject_t, int, unsigned char);
__device__ void __sust_b_1d_i16_trap(cudaSurfaceObject_t, int, unsigned short);
__device__ void __sust_b_1d_i32_trap(cudaSurfaceObject_t, int, unsigned int);
__device__ void __sust_b_1d_i64_trap(cudaSurfaceObject_t, int, unsigned long long);
__device__ void __sust_b_1d_v2i8_trap(cudaSurfaceObject_t, int, uchar2);
__device__ void __sust_b_1d_v2i16_trap(cudaSurfaceObject_t, int, ushort2);
__device__ void __sust_b_1d_v2i32_trap(cudaSurfaceObject_t, int, uint2);
__device__ void __sust_b_1d_v2i64_trap(cudaSurfaceObject_t, int, ulonglong2);
__device__ void __sust_b_1d_v4i8_trap(cudaSurfaceObject_t, int, uchar4);
__device__ void __sust_b_1d_v4i16_trap(cudaSurfaceObject_t, int, ushort4);
__device__ void __sust_b_1d_v4i32_trap(cudaSurfaceObject_t, int, uint4);
__device__ void __sust_b_1d_array_i8_trap(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_1d_array_i16_trap(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_1d_array_i32_trap(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_1d_array_i64_trap(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_1d_array_v2i8_trap(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_1d_array_v2i16_trap(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_1d_array_v2i32_trap(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_1d_array_v2i64_trap(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_1d_array_v4i8_trap(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_1d_array_v4i16_trap(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_1d_array_v4i32_trap(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_i8_trap(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_2d_i16_trap(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_2d_i32_trap(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_2d_i64_trap(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_2d_v2i8_trap(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_2d_v2i16_trap(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_2d_v2i32_trap(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_2d_v2i64_trap(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_2d_v4i8_trap(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_2d_v4i16_trap(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_2d_v4i32_trap(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_array_i8_trap(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_2d_array_i16_trap(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_2d_array_i32_trap(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_2d_array_i64_trap(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_2d_array_v2i8_trap(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_2d_array_v2i16_trap(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_2d_array_v2i32_trap(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_2d_array_v2i64_trap(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_2d_array_v4i8_trap(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_2d_array_v4i16_trap(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_2d_array_v4i32_trap(cudaSurfaceObject_t, int, int, int, uint4);
__device__ void __sust_b_3d_i8_trap(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_3d_i16_trap(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_3d_i32_trap(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_3d_i64_trap(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_3d_v2i8_trap(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_3d_v2i16_trap(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_3d_v2i32_trap(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_3d_v2i64_trap(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_3d_v4i8_trap(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_3d_v4i16_trap(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_3d_v4i32_trap(cudaSurfaceObject_t, int, int, int, uint4);
__device__ void __sust_b_1d_i8_zero(cudaSurfaceObject_t, int, unsigned char);
__device__ void __sust_b_1d_i16_zero(cudaSurfaceObject_t, int, unsigned short);
__device__ void __sust_b_1d_i32_zero(cudaSurfaceObject_t, int, unsigned int);
__device__ void __sust_b_1d_i64_zero(cudaSurfaceObject_t, int, unsigned long long);
__device__ void __sust_b_1d_v2i8_zero(cudaSurfaceObject_t, int, uchar2);
__device__ void __sust_b_1d_v2i16_zero(cudaSurfaceObject_t, int, ushort2);
__device__ void __sust_b_1d_v2i32_zero(cudaSurfaceObject_t, int, uint2);
__device__ void __sust_b_1d_v2i64_zero(cudaSurfaceObject_t, int, ulonglong2);
__device__ void __sust_b_1d_v4i8_zero(cudaSurfaceObject_t, int, uchar4);
__device__ void __sust_b_1d_v4i16_zero(cudaSurfaceObject_t, int, ushort4);
__device__ void __sust_b_1d_v4i32_zero(cudaSurfaceObject_t, int, uint4);
__device__ void __sust_b_1d_array_i8_zero(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_1d_array_i16_zero(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_1d_array_i32_zero(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_1d_array_i64_zero(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_1d_array_v2i8_zero(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_1d_array_v2i16_zero(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_1d_array_v2i32_zero(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_1d_array_v2i64_zero(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_1d_array_v4i8_zero(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_1d_array_v4i16_zero(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_1d_array_v4i32_zero(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_i8_zero(cudaSurfaceObject_t, int, int, unsigned char);
__device__ void __sust_b_2d_i16_zero(cudaSurfaceObject_t, int, int, unsigned short);
__device__ void __sust_b_2d_i32_zero(cudaSurfaceObject_t, int, int, unsigned int);
__device__ void __sust_b_2d_i64_zero(cudaSurfaceObject_t, int, int, unsigned long long);
__device__ void __sust_b_2d_v2i8_zero(cudaSurfaceObject_t, int, int, uchar2);
__device__ void __sust_b_2d_v2i16_zero(cudaSurfaceObject_t, int, int, ushort2);
__device__ void __sust_b_2d_v2i32_zero(cudaSurfaceObject_t, int, int, uint2);
__device__ void __sust_b_2d_v2i64_zero(cudaSurfaceObject_t, int, int, ulonglong2);
__device__ void __sust_b_2d_v4i8_zero(cudaSurfaceObject_t, int, int, uchar4);
__device__ void __sust_b_2d_v4i16_zero(cudaSurfaceObject_t, int, int, ushort4);
__device__ void __sust_b_2d_v4i32_zero(cudaSurfaceObject_t, int, int, uint4);
__device__ void __sust_b_2d_array_i8_zero(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_2d_array_i16_zero(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_2d_array_i32_zero(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_2d_array_i64_zero(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_2d_array_v2i8_zero(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_2d_array_v2i16_zero(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_2d_array_v2i32_zero(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_2d_array_v2i64_zero(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_2d_array_v4i8_zero(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_2d_array_v4i16_zero(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_2d_array_v4i32_zero(cudaSurfaceObject_t, int, int, int, uint4);
__device__ void __sust_b_3d_i8_zero(cudaSurfaceObject_t, int, int, int, unsigned char);
__device__ void __sust_b_3d_i16_zero(cudaSurfaceObject_t, int, int, int, unsigned short);
__device__ void __sust_b_3d_i32_zero(cudaSurfaceObject_t, int, int, int, unsigned int);
__device__ void __sust_b_3d_i64_zero(cudaSurfaceObject_t, int, int, int, unsigned long long);
__device__ void __sust_b_3d_v2i8_zero(cudaSurfaceObject_t, int, int, int, uchar2);
__device__ void __sust_b_3d_v2i16_zero(cudaSurfaceObject_t, int, int, int, ushort2);
__device__ void __sust_b_3d_v2i32_zero(cudaSurfaceObject_t, int, int, int, uint2);
__device__ void __sust_b_3d_v2i64_zero(cudaSurfaceObject_t, int, int, int, ulonglong2);
__device__ void __sust_b_3d_v4i8_zero(cudaSurfaceObject_t, int, int, int, uchar4);
__device__ void __sust_b_3d_v4i16_zero(cudaSurfaceObject_t, int, int, int, ushort4);
__device__ void __sust_b_3d_v4i32_zero(cudaSurfaceObject_t, int, int, int, uint4);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surf1Dread(&ret, surfObject, x, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surf2Dread(&ret, surfObject, x, y, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* 3D Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surf3Dread(&ret, surfObject, x, y, z, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surfCubemapread(&ret, surfObject, face, x, y, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__SURFACE_INDIRECT_FUNCTIONS_DECL__ T surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
   T ret;
   surfCubemapLayeredread(&ret, surfObject, x, y, z, layerface, boundaryMode);
   return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
* 2D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);


/*******************************************************************************
*                                                                              *
* 3D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);

#endif // __CUDA_ARCH__ || __CUDA_ARCH__ >= 200

#endif // __cplusplus && __CUDACC__

#undef __SURFACE_INDIRECT_FUNCTIONS_DECL__

#if !defined(__CUDACC_RTC__)
#include "surface_indirect_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#endif // __SURFACE_INDIRECT_FUNCTIONS_H__


