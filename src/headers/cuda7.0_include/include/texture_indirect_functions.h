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


#ifndef __TEXTURE_INDIRECT_FUNCTIONS_H__
#define __TEXTURE_INDIRECT_FUNCTIONS_H__

#if defined(__CUDACC_RTC__)
#define __TEXTURE_INDIRECT_FUNCTIONS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __TEXTURE_INDIRECT_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200


#include "builtin_types.h"
#include "host_defines.h"
#include "vector_functions.h"

extern "C" {
__device__ void __tex_1d_v4f32_s32(cudaTextureObject_t, int, float *, float *, float *, float *);
__device__ void __tex_1d_v4f32_f32(cudaTextureObject_t, float, float *, float *, float *, float *);
__device__ void __tex_1d_level_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tex_1d_grad_v4f32_f32(cudaTextureObject_t, float, float, float, float *, float *, float *, float *);
__device__ void __tex_1d_v4s32_s32(cudaTextureObject_t, int, int *, int *, int *, int *);
__device__ void __tex_1d_v4s32_f32(cudaTextureObject_t, float, int *, int *, int *, int *);
__device__ void __tex_1d_level_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tex_1d_grad_v4s32_f32(cudaTextureObject_t, float, float, float, int *, int *, int *, int *);
__device__ void __tex_1d_v4u32_s32(cudaTextureObject_t, int, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_v4u32_f32(cudaTextureObject_t, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_level_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_grad_v4u32_f32(cudaTextureObject_t, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_1d_array_v4f32_s32(cudaTextureObject_t, int, int, float *, float *, float *, float *);
__device__ void __tex_1d_array_v4f32_f32(cudaTextureObject_t, int, float, float *, float *, float *, float *);
__device__ void __tex_1d_array_level_v4f32_f32(cudaTextureObject_t, int, float, float, float *, float *, float *, float *);
__device__ void __tex_1d_array_grad_v4f32_f32(cudaTextureObject_t, int, float, float, float, float *, float *, float *, float *);
__device__ void __tex_1d_array_v4s32_s32(cudaTextureObject_t, int, int, int *, int *, int *, int *);
__device__ void __tex_1d_array_v4s32_f32(cudaTextureObject_t, int, float, int *, int *, int *, int *);
__device__ void __tex_1d_array_level_v4s32_f32(cudaTextureObject_t, int, float, float, int *, int *, int *, int *);
__device__ void __tex_1d_array_grad_v4s32_f32(cudaTextureObject_t, int, float, float, float, int *, int *, int *, int *);
__device__ void __tex_1d_array_v4u32_s32(cudaTextureObject_t, int, int, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_array_v4u32_f32(cudaTextureObject_t, int, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_array_level_v4u32_f32(cudaTextureObject_t, int, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_1d_array_grad_v4u32_f32(cudaTextureObject_t, int, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_2d_v4f32_s32(cudaTextureObject_t, int, int, float *, float *, float *, float *);
__device__ void __tex_2d_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_level_v4f32_f32(cudaTextureObject_t, float, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_grad_v4f32_f32(cudaTextureObject_t, float, float, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_v4s32_s32(cudaTextureObject_t, int, int, int *, int *, int *, int *);
__device__ void __tex_2d_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_level_v4s32_f32(cudaTextureObject_t, float, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_grad_v4s32_f32(cudaTextureObject_t, float, float, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_v4u32_s32(cudaTextureObject_t, int, int, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_level_v4u32_f32(cudaTextureObject_t, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_grad_v4u32_f32(cudaTextureObject_t, float, float, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_2d_array_v4f32_s32(cudaTextureObject_t, int, int, int, float *, float *, float *, float *);
__device__ void __tex_2d_array_v4f32_f32(cudaTextureObject_t, int, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_array_level_v4f32_f32(cudaTextureObject_t, int, float, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_array_grad_v4f32_f32(cudaTextureObject_t, int, float, float, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_2d_array_v4s32_s32(cudaTextureObject_t, int, int, int, int *, int *, int *, int *);
__device__ void __tex_2d_array_v4s32_f32(cudaTextureObject_t, int, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_array_level_v4s32_f32(cudaTextureObject_t, int, float, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_array_grad_v4s32_f32(cudaTextureObject_t, int, float, float, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_2d_array_v4u32_s32(cudaTextureObject_t, int, int, int, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_array_v4u32_f32(cudaTextureObject_t, int, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_array_level_v4u32_f32(cudaTextureObject_t, int, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_2d_array_grad_v4u32_f32(cudaTextureObject_t, int, float, float, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_3d_v4f32_s32(cudaTextureObject_t, int, int, int, float *, float *, float *, float *);
__device__ void __tex_3d_v4f32_f32(cudaTextureObject_t, float, float, float, float *, float *, float *, float *);
__device__ void __tex_3d_level_v4f32_f32(cudaTextureObject_t, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_3d_grad_v4f32_f32(cudaTextureObject_t, float, float, float, float, float, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_3d_v4s32_s32(cudaTextureObject_t, int, int, int, int *, int *, int *, int *);
__device__ void __tex_3d_v4s32_f32(cudaTextureObject_t, float, float, float, int *, int *, int *, int *);
__device__ void __tex_3d_level_v4s32_f32(cudaTextureObject_t, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_3d_grad_v4s32_f32(cudaTextureObject_t, float, float, float, float, float, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_3d_v4u32_s32(cudaTextureObject_t, int, int, int, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_3d_v4u32_f32(cudaTextureObject_t, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_3d_level_v4u32_f32(cudaTextureObject_t, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_3d_grad_v4u32_f32(cudaTextureObject_t, float, float, float, float, float, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_cube_v4f32_f32(cudaTextureObject_t, float, float, float, float *, float *, float *, float *);
__device__ void __tex_cube_level_v4f32_f32(cudaTextureObject_t, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_cube_v4s32_f32(cudaTextureObject_t, float, float, float, int *, int *, int *, int *);
__device__ void __tex_cube_level_v4s32_f32(cudaTextureObject_t, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_cube_v4u32_f32(cudaTextureObject_t, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_cube_level_v4u32_f32(cudaTextureObject_t, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tex_cube_array_v4f32_f32(cudaTextureObject_t, int, float, float, float, float *, float *, float *, float *);
__device__ void __tex_cube_array_level_v4f32_f32(cudaTextureObject_t, int, float, float, float, float, float *, float *, float *, float *);
__device__ void __tex_cube_array_v4s32_f32(cudaTextureObject_t, int, float, float, float, int *, int *, int *, int *);
__device__ void __tex_cube_array_level_v4s32_f32(cudaTextureObject_t, int, float, float, float, float, int *, int *, int *, int *);
__device__ void __tex_cube_array_v4u32_f32(cudaTextureObject_t, int, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tex_cube_array_level_v4u32_f32(cudaTextureObject_t, int, float, float, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

__device__ void __tld4_r_2d_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tld4_g_2d_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tld4_b_2d_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tld4_a_2d_v4f32_f32(cudaTextureObject_t, float, float, float *, float *, float *, float *);
__device__ void __tld4_r_2d_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tld4_g_2d_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tld4_b_2d_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tld4_a_2d_v4s32_f32(cudaTextureObject_t, float, float, int *, int *, int *, int *);
__device__ void __tld4_r_2d_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tld4_g_2d_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tld4_b_2d_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
__device__ void __tld4_a_2d_v4u32_f32(cudaTextureObject_t, float, float, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
}

/*******************************************************************************
*                                                                              *
* 1D Linear Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char *retVal, cudaTextureObject_t texObject, int x);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(signed char *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char4 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned char *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar4 *retVal, cudaTextureObject_t texObject, int x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short4 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned short *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort4 *retVal, cudaTextureObject_t texObject, int x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int4 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned int *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint4 *retVal, cudaTextureObject_t texObject, int x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long4 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned long *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong4 *retVal, cudaTextureObject_t texObject, int x);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float1 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float2 *retVal, cudaTextureObject_t texObject, int x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float4 *retVal, cudaTextureObject_t texObject, int x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1Dfetch(cudaTextureObject_t texObject, int x)
{
  T ret;
  tex1Dfetch(&ret, texObject, x);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char *retVal, cudaTextureObject_t texObject, float x);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(signed char *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char4 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned char *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar4 *retVal, cudaTextureObject_t texObject, float x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short4 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned short *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort4 *retVal, cudaTextureObject_t texObject, float x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int4 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned int *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint4 *retVal, cudaTextureObject_t texObject, float x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long4 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned long *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong4 *retVal, cudaTextureObject_t texObject, float x);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float1 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float2 *retVal, cudaTextureObject_t texObject, float x);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float4 *retVal, cudaTextureObject_t texObject, float x);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1D(cudaTextureObject_t texObject, float x)
{
  T ret;
  tex1D(&ret, texObject, x);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char *retVal, cudaTextureObject_t texObject, float x, float y);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(signed char *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char4 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short4 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int4 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint4 *retVal, cudaTextureObject_t texObject, float x, float y);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long4 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float1 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float2 *retVal, cudaTextureObject_t texObject, float x, float y);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float4 *retVal, cudaTextureObject_t texObject, float x, float y);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2D(cudaTextureObject_t texObject, float x, float y)
{
  T ret;
  tex2D(&ret, texObject, x, y);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 3D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char *retVal, cudaTextureObject_t texObject, float x, float y, float z);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex3D(cudaTextureObject_t texObject, float x, float y, float z)
{
  T ret;
  tex3D(&ret, texObject, x, y, z);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char *retVal, cudaTextureObject_t texObject, float x, int layer);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(signed char *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char4 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned char *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar4 *retVal, cudaTextureObject_t texObject, float x, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short4 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned short *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort4 *retVal, cudaTextureObject_t texObject, float x, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int4 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned int *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint4 *retVal, cudaTextureObject_t texObject, float x, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long4 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned long *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong4 *retVal, cudaTextureObject_t texObject, float x, int layer);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float1 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float2 *retVal, cudaTextureObject_t texObject, float x, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float4 *retVal, cudaTextureObject_t texObject, float x, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1DLayered(cudaTextureObject_t texObject, float x, int layer)
{
  T ret;
  tex1DLayered(&ret, texObject, x, layer);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char *retVal, cudaTextureObject_t texObject, float x, float y, int layer);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(signed char *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer)
{
  T ret;
  tex2DLayered(&ret, texObject, x, y, layer);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(char *retVal, cudaTextureObject_t texObject, float x, float y, float z);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(short *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(int *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(long *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(float *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemap(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T texCubemap(cudaTextureObject_t texObject, float x, float y, float z)
{
  T ret;
  texCubemap(&ret, texObject, x, y, z);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(short *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(int *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(long *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(float *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayered(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer)
{
  T ret;
  texCubemapLayered(&ret, texObject, x, y, z, layer);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Texture indirect gather functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(signed char *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float1 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float2 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float4 *retVal, cudaTextureObject_t texObject, float x, float y, int comp = 0);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0)
{
  T ret;
  tex2Dgather(&ret, to, x, y, comp);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char *retVal, cudaTextureObject_t texObject, float x, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(signed char *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char4 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short4 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int4 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long4 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float1 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float2 *retVal, cudaTextureObject_t texObject, float x, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float4 *retVal, cudaTextureObject_t texObject, float x, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1DLod(cudaTextureObject_t texObject, float x, float level)
{
  T ret;
  tex1DLod(&ret, texObject, x, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char *retVal, cudaTextureObject_t texObject, float x, float y, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level)
{
  T ret;
  tex2DLod(&ret, texObject, x, y, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 3D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
  T ret;
  tex3DLod(&ret, texObject, x, y, z, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char *retVal, cudaTextureObject_t texObject, float x, int layer, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(signed char *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float1 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float2 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float4 *retVal, cudaTextureObject_t texObject, float x, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level)
{
  T ret;
  tex1DLayeredLod(&ret, texObject, x, layer, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(signed char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level)
{
  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(float *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLod(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
  T ret;
  texCubemapLod(&ret, texObject, x, y, z, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(short *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(int *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(long *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(float *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texCubemapLayeredLod(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, int layer, float level);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  T ret;
  texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(signed char *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned char *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned short *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned int *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned long *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float1 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float2 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float4 *retVal, cudaTextureObject_t texObject, float x, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  T ret;
  tex1DGrad(&ret, texObject, x, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 3D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(signed char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float1 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float2 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 1D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(signed char *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned char *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned short *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned int *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned long *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float1 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float2 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float4 *retVal, cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  T ret;
  tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy);
  return ret;
}

/*******************************************************************************
*                                                                              *
* 2D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(signed char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned char *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned short *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned int *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned long *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float1 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float2 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float4 *retVal, cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <class T>
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy);
  return ret;
}

#endif // __CUDA_ARCH__ || __CUDA_ARCH__ >= 200

#endif // __cplusplus && __CUDACC__

#if !defined(__CUDACC_RTC__)
#include "texture_indirect_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#undef __TEXTURE_INDIRECT_FUNCTIONS_DECL__

#endif // __TEXTURE_INDIRECT_FUNCTIONS_H__


