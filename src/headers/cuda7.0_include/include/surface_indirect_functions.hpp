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


#ifndef __SURFACE_INDIRECT_FUNCTIONS_HPP__
#define __SURFACE_INDIRECT_FUNCTIONS_HPP__

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

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i8_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i8_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i8_zero(&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i8_trap((char *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i8_clamp((char *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i8_zero((char *)&tmp, surfObject, x);
  }
  *retVal = (signed char)(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  char1 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i8_trap((char *)&tmp.x, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i8_clamp((char *)&tmp.x, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i8_zero((char *)&tmp.x, surfObject, x);
  }
  *retVal = make_char1(tmp.x);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i8_trap((char *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i8_clamp((char *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i8_zero((char *)&tmp, surfObject, x);
  }
  *retVal = (unsigned char)(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar1 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i8_trap((char *)&tmp.x, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i8_clamp((char *)&tmp.x, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i8_zero((char *)&tmp.x, surfObject, x);
  }
  *retVal = make_uchar1(tmp.x);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i16_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i16_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i16_zero(&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i16_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i16_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i16_zero(&tmp, surfObject, x);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i16_trap((short *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i16_clamp((short *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i16_zero((short *)&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i16_trap((short *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i16_clamp((short *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i16_zero((short *)&tmp, surfObject, x);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero(&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero(&tmp, surfObject, x);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero((int *)&tmp, surfObject, x);
  }
  *retVal = (unsigned int)(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero((int *)&tmp, surfObject, x);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i64_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i64_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i64_zero(&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i64_trap(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i64_clamp(&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i64_zero(&tmp, surfObject, x);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i64_trap((long long *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i64_clamp((long long *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i64_zero((long long *)&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i64_trap((long long *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i64_clamp((long long *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i64_zero((long long *)&tmp, surfObject, x);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero((int *)&tmp, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_i32_trap((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_i32_clamp((int *)&tmp, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_i32_zero((int *)&tmp, surfObject, x);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* 2D Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i8_trap(&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i8_clamp(&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i8_zero(&tmp, surfObject, x, y);
  }
  *retVal = (char)(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i8_trap((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i8_clamp((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i8_zero((char *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i8_trap((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i8_clamp((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i8_zero((char *)&tmp, surfObject, x, y);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i8_trap((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i8_clamp((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i8_zero((char *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i8_trap((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i8_clamp((char *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i8_zero((char *)&tmp, surfObject, x, y);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i16_trap((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i16_clamp((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i16_zero((short *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i16_trap((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i16_clamp((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i16_zero((short *)&tmp, surfObject, x, y);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i16_trap((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i16_clamp((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i16_zero((short *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i16_trap((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i16_clamp((short *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i16_zero((short *)&tmp, surfObject, x, y);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i64_trap((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i64_clamp((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i64_zero((long long *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i64_trap((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i64_clamp((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i64_zero((long long *)&tmp, surfObject, x, y);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i64_trap((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i64_clamp((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i64_zero((long long *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i64_trap((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i64_clamp((long long *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i64_zero((long long *)&tmp, surfObject, x, y);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_i32_trap((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_i32_clamp((int *)&tmp, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_i32_zero((int *)&tmp, surfObject, x, y);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* 3D Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i8_trap((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i8_clamp((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i8_zero((char *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i8_trap((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i8_clamp((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i8_zero((char *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i8_trap((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i8_clamp((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i8_zero((char *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i8_trap((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i8_clamp((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i8_zero((char *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i8_trap((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i8_clamp((char *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i8_zero((char *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i16_trap((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i16_clamp((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i16_zero((short *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i16_trap((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i16_clamp((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i16_zero((short *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i16_trap((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i16_clamp((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i16_zero((short *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i16_trap((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i16_clamp((short *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i16_zero((short *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i64_trap((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i64_clamp((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i64_zero((long long *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i64_trap((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i64_clamp((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i64_zero((long long *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i64_trap((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i64_clamp((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i64_zero((long long *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i64_trap((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i64_clamp((long long *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i64_zero((long long *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_i32_trap((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_i32_clamp((int *)&tmp, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_i32_zero((int *)&tmp, surfObject, x, y, z);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, x, y, z);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_3d_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_3d_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_3d_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, x, y, z);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i8_trap((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i8_clamp((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i8_zero((char *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i8_trap((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i8_clamp((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i8_zero((char *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i8_trap((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i8_clamp((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i8_zero((char *)&tmp, surfObject, layer, x);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i8_trap((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i8_clamp((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i8_zero((char *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i8_trap((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i8_clamp((char *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i8_zero((char *)&tmp, surfObject, layer, x);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i16_trap((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i16_clamp((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i16_zero((short *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i16_trap((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i16_clamp((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i16_zero((short *)&tmp, surfObject, layer, x);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i16_trap((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i16_clamp((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i16_zero((short *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i16_trap((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i16_clamp((short *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i16_zero((short *)&tmp, surfObject, layer, x);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i64_trap((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i64_clamp((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i64_zero((long long *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i64_trap((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i64_clamp((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i64_zero((long long *)&tmp, surfObject, layer, x);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i64_trap((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i64_clamp((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i64_zero((long long *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i64_trap((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i64_clamp((long long *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i64_zero((long long *)&tmp, surfObject, layer, x);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_i32_trap((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_i32_clamp((int *)&tmp, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_i32_zero((int *)&tmp, surfObject, layer, x);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_1d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_1d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_1d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layer, x, y);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layer, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, face, x, y);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, face, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, face, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect read functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(signed char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  signed char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_char1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned char *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned char tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i8_trap((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i8_clamp((char *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i8_zero((char *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_uchar1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_short1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned short *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned short tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i16_trap((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i16_clamp((short *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i16_zero((short *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_ushort1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_int1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned int *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned int tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_uint1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(longlong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_longlong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(unsigned long long *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ulonglong1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  unsigned long long tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i64_trap((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i64_clamp((long long *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i64_zero((long long *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_ulonglong1(tmp);
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float1 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  float tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_i32_trap((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_i32_clamp((int *)&tmp, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_i32_zero((int *)&tmp, surfObject, layerface, x, y);
  }
  *retVal = make_float1(tmp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  char2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i8_trap((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i8_clamp((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i8_zero((char *)&tmp.x, (char *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  short2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i16_trap((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i16_clamp((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i16_zero((short *)&tmp.x, (short *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  int2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  uint2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(longlong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  longlong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ulonglong2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  ulonglong2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i64_trap((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i64_clamp((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i64_zero((long long *)&tmp.x, (long long *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float2 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  float2 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v2i32_trap((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v2i32_clamp((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v2i32_zero((int *)&tmp.x, (int *)&tmp.y, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(char4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  char4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uchar4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  uchar4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i8_trap((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i8_clamp((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i8_zero((char *)&tmp.x, (char *)&tmp.y, (char *)&tmp.z, (char *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(short4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  short4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(ushort4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  ushort4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i16_trap((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i16_clamp((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i16_zero((short *)&tmp.x, (short *)&tmp.y, (short *)&tmp.z, (short *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(int4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  int4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(uint4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  uint4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredread(float4 *retVal, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  float4 tmp;
  if (boundaryMode == cudaBoundaryModeTrap) {
    __suld_2d_array_v4i32_trap((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __suld_2d_array_v4i32_clamp((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __suld_2d_array_v4i32_zero((int *)&tmp.x, (int *)&tmp.y, (int *)&tmp.z, (int *)&tmp.w, surfObject, layerface, x, y);
  }
  *retVal = tmp;
}

/*******************************************************************************
*                                                                              *
* 1D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i8_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i8_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i8_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i8_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i8_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i8_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i8_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i8_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i8_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i8_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i8_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i8_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i8_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i8_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i8_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i16_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i16_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i16_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i16_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i16_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i16_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i16_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i16_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i16_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i16_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i16_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i16_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i64_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i64_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i64_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i64_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i64_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i64_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i64_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i64_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i64_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i64_trap(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i64_clamp(surfObject, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i64_zero(surfObject, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_i32_trap(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_i32_clamp(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_i32_zero(surfObject, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i8_trap(surfObject, x, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i8_clamp(surfObject, x, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i8_zero(surfObject, x, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i8_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i8_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i8_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i16_trap(surfObject, x, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i16_clamp(surfObject, x, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i16_zero(surfObject, x, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i16_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i16_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i16_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i32_trap(surfObject, x, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i32_clamp(surfObject, x, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i32_zero(surfObject, x, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i32_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i32_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i32_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i64_trap(surfObject, x, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i64_clamp(surfObject, x, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i64_zero(surfObject, x, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i64_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i64_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i64_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v2i32_trap(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v2i32_clamp(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v2i32_zero(surfObject, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i8_trap(surfObject, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i8_clamp(surfObject, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i8_zero(surfObject, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i8_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i8_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i8_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i16_trap(surfObject, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i16_clamp(surfObject, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i16_zero(surfObject, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i16_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i16_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i16_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i32_trap(surfObject, x, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i32_clamp(surfObject, x, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i32_zero(surfObject, x, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i32_trap(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i32_clamp(surfObject, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i32_zero(surfObject, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_v4i32_trap(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_v4i32_clamp(surfObject, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_v4i32_zero(surfObject, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
* 2D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i8_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i8_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i8_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i8_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i8_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i8_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i8_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i8_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i8_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i8_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i8_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i8_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i8_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i8_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i8_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i16_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i16_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i16_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i16_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i16_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i16_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i16_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i16_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i16_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i16_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i16_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i16_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i64_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i64_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i64_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i64_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i64_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i64_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i64_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i64_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i64_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i64_trap(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i64_clamp(surfObject, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i64_zero(surfObject, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_i32_trap(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_i32_clamp(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_i32_zero(surfObject, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i8_trap(surfObject, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i8_clamp(surfObject, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i8_zero(surfObject, x, y, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i8_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i8_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i8_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i16_trap(surfObject, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i16_clamp(surfObject, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i16_zero(surfObject, x, y, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i16_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i16_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i16_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i32_trap(surfObject, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i32_clamp(surfObject, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i32_zero(surfObject, x, y, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i32_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i32_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i32_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i64_trap(surfObject, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i64_clamp(surfObject, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i64_zero(surfObject, x, y, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i64_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i64_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i64_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v2i32_trap(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v2i32_clamp(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v2i32_zero(surfObject, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i8_trap(surfObject, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i8_clamp(surfObject, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i8_zero(surfObject, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i8_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i8_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i8_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i16_trap(surfObject, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i16_clamp(surfObject, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i16_zero(surfObject, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i16_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i16_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i16_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i32_trap(surfObject, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i32_clamp(surfObject, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i32_zero(surfObject, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i32_trap(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i32_clamp(surfObject, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i32_zero(surfObject, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_v4i32_trap(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_v4i32_clamp(surfObject, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_v4i32_zero(surfObject, x, y, cvt.i);
  }
}


/*******************************************************************************
*                                                                              *
* 3D Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i8_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i8_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i8_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i8_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i8_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i8_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i8_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i8_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i8_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i8_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i8_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i8_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i8_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i8_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i8_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i16_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i16_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i16_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i16_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i16_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i16_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i16_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i16_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i16_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i16_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i16_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i16_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i64_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i64_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i64_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i64_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i64_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i64_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i64_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i64_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i64_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i64_trap(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i64_clamp(surfObject, x, y, z, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i64_zero(surfObject, x, y, z, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_i32_trap(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_i32_clamp(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_i32_zero(surfObject, x, y, z, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i8_trap(surfObject, x, y, z, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i8_clamp(surfObject, x, y, z, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i8_zero(surfObject, x, y, z, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i8_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i8_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i8_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i16_trap(surfObject, x, y, z, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i16_clamp(surfObject, x, y, z, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i16_zero(surfObject, x, y, z, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i16_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i16_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i16_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i32_trap(surfObject, x, y, z, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i32_clamp(surfObject, x, y, z, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i32_zero(surfObject, x, y, z, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i32_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i32_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i32_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i64_trap(surfObject, x, y, z, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i64_clamp(surfObject, x, y, z, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i64_zero(surfObject, x, y, z, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i64_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i64_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i64_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v2i32_trap(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v2i32_clamp(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v2i32_zero(surfObject, x, y, z, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i8_trap(surfObject, x, y, z, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i8_clamp(surfObject, x, y, z, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i8_zero(surfObject, x, y, z, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i8_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i8_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i8_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i16_trap(surfObject, x, y, z, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i16_clamp(surfObject, x, y, z, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i16_zero(surfObject, x, y, z, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i16_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i16_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i16_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i32_trap(surfObject, x, y, z, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i32_clamp(surfObject, x, y, z, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i32_zero(surfObject, x, y, z, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i32_trap(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i32_clamp(surfObject, x, y, z, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i32_zero(surfObject, x, y, z, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf3Dwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_3d_v4i32_trap(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_3d_v4i32_clamp(surfObject, x, y, z, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_3d_v4i32_zero(surfObject, x, y, z, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
* 1D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i8_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i8_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i8_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i8_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i8_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i8_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i8_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i8_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i8_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i8_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i8_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i8_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i8_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i8_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i8_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i16_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i16_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i16_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i16_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i16_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i16_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i16_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i16_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i16_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i16_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i16_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i16_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i64_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i64_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i64_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i64_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i64_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i64_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i64_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i64_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i64_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i64_trap(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i64_clamp(surfObject, layer, x, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i64_zero(surfObject, layer, x, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_i32_trap(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_i32_clamp(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_i32_zero(surfObject, layer, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i8_trap(surfObject, layer, x, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i8_clamp(surfObject, layer, x, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i8_zero(surfObject, layer, x, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i8_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i8_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i8_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i16_trap(surfObject, layer, x, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i16_clamp(surfObject, layer, x, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i16_zero(surfObject, layer, x, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i16_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i16_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i16_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i32_trap(surfObject, layer, x, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i32_clamp(surfObject, layer, x, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i32_zero(surfObject, layer, x, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i32_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i32_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i32_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i64_trap(surfObject, layer, x, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i64_clamp(surfObject, layer, x, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i64_zero(surfObject, layer, x, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i64_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i64_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i64_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v2i32_trap(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v2i32_clamp(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v2i32_zero(surfObject, layer, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i8_trap(surfObject, layer, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i8_clamp(surfObject, layer, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i8_zero(surfObject, layer, x, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i8_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i8_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i8_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i16_trap(surfObject, layer, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i16_clamp(surfObject, layer, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i16_zero(surfObject, layer, x, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i16_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i16_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i16_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i32_trap(surfObject, layer, x, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i32_clamp(surfObject, layer, x, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i32_zero(surfObject, layer, x, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i32_trap(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i32_clamp(surfObject, layer, x, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i32_zero(surfObject, layer, x, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf1DLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_1d_array_v4i32_trap(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_1d_array_v4i32_clamp(surfObject, layer, x, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_1d_array_v4i32_zero(surfObject, layer, x, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
* 2D Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layer, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layer, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layer, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, layer, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, layer, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, layer, x, y, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, layer, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, layer, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, layer, x, y, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layer, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layer, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layer, x, y, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, layer, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, layer, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, layer, x, y, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layer, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, layer, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, layer, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, layer, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, layer, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, layer, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, layer, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layer, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layer, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layer, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layer, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layer, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surf2DLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layer, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layer, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
* Cubemap Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, face, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, face, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, face, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, face, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, face, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, face, x, y, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, face, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, face, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, face, x, y, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, face, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, face, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, face, x, y, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, face, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, face, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, face, x, y, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, face, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, face, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, face, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, face, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, face, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, face, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, face, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, face, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, face, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, face, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, face, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, face, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, face, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, face, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
* Cubemap Layered Surface indirect write functions
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(signed char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned char data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i8_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i8_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i8_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned short data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i16_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i16_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i16_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned int data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long long data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i64_trap(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i64_clamp(surfObject, layerface, x, y, data.x);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i64_zero(surfObject, layerface, x, y, data.x);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, cvt.i);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float1 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float f; unsigned int i; } cvt;
  cvt.f = data.x;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_i32_trap(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_i32_clamp(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_i32_zero(surfObject, layerface, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, layerface, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, layerface, x, y, make_uchar2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, layerface, x, y, make_uchar2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i8_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i8_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i8_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, layerface, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, layerface, x, y, make_ushort2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, layerface, x, y, make_ushort2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i16_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i16_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i16_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layerface, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layerface, x, y, make_uint2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layerface, x, y, make_uint2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, layerface, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, layerface, x, y, make_ulonglong2(data.x, data.y));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, layerface, x, y, make_ulonglong2(data.x, data.y));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i64_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i64_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i64_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float2 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float2 f; uint2 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v2i32_trap(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v2i32_clamp(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v2i32_zero(surfObject, layerface, x, y, cvt.i);
  }
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, layerface, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, layerface, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, layerface, x, y, make_uchar4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i8_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i8_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i8_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, layerface, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, layerface, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, layerface, x, y, make_ushort4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i16_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i16_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i16_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layerface, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layerface, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layerface, x, y, make_uint4(data.x, data.y, data.z, data.w));
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layerface, x, y, data);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layerface, x, y, data);
  }
}

__SURFACE_INDIRECT_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float4 data, cudaSurfaceObject_t surfObject, int x, int y, int z, int layerface, cudaSurfaceBoundaryMode boundaryMode)
{
  union { float4 f; uint4 i; } cvt;
  cvt.f = data;

  if (boundaryMode == cudaBoundaryModeTrap) {
    __sust_b_2d_array_v4i32_trap(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeClamp) {
    __sust_b_2d_array_v4i32_clamp(surfObject, layerface, x, y, cvt.i);
  }
  else if (boundaryMode == cudaBoundaryModeZero) {
    __sust_b_2d_array_v4i32_zero(surfObject, layerface, x, y, cvt.i);
  }
}

#endif // __CUDA_ARCH__ || __CUDA_ARCH__ >= 200

#endif // __cplusplus && __CUDACC__

#undef __SURFACE_INDIRECT_FUNCTIONS_DECL__

#endif // __SURFACE_INDIRECT_FUNCTIONS_HPP__



