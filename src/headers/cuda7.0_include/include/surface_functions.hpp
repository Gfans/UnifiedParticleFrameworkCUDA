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

#if !defined(__SURFACE_FUNCTIONS_HPP__)
#define __SURFACE_FUNCTIONS_HPP__

#if defined(__CUDACC_RTC__)
#define __SURFACE_FUNCTIONS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SURFACE_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "cuda_surface_types.h"
#include "host_defines.h"
#include "surface_types.h"
#include "vector_functions.h"
#include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeClamp) : \
                                           __surf1Dwrite##type(val, surf, x, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, mode, type) \
        __surf1Dwrite##type(val, surf, x, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(signed char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeClamp) : \
                                           __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, mode, type) \
        __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(signed char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, z, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeClamp) : \
                                           __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, z, mode, type) \
        __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(signed char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, z, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, z, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, z, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, z, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, z, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, z, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeClamp) : \
                                           __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, layer, mode, type) \
        __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(signed char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, layer, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, layer, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, layer, mode, type) \
        __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(signed char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, layer, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, layer, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, face, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, face, mode, type) \
        __surf2DLayeredwrite##type(val, surf, x, y, face, cudaBoundaryModeTrap)


#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(signed char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, face, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, face, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, face, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, face, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, face, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, face, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, layerFace, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, layerFace, mode, type) \
       __surf2DLayeredwrite##type(val, surf, x, y, layerFace, cudaBoundaryModeTrap)


#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(signed char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, layerFace, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, layerFace, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, layerFace, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, layerFace, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, layerFace, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, layerFace, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* __cplusplus && __CUDACC__ */

#undef __SURFACE_FUNCTIONS_DECL__

#endif /* !__SURFACE_FUNCTIONS_HPP__ */

