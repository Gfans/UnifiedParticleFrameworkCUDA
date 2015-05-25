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

#if !defined(__SURFACE_FUNCTIONS_H__)
#define __SURFACE_FUNCTIONS_H__

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
extern __device__ __device_builtin__ uchar1     __surf1Dreadc1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar2     __surf1Dreadc2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar4     __surf1Dreadc4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort1    __surf1Dreads1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort2    __surf1Dreads2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort4    __surf1Dreads4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint1      __surf1Dreadu1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint2      __surf1Dreadu2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint4      __surf1Dreadu4(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong1 __surf1Dreadl1(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong2 __surf1Dreadl2(surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf1Dread##type(surf, x, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1Dread##type(surf, x, cudaBoundaryModeClamp) : \
                                           __surf1Dread##type(surf, x, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, mode, type) \
         __surf1Dread##type(surf, x, cudaBoundaryModeTrap)

#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surf1Dread(&tmp, surf, x, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surf1Dread<T>(surf, x, mode);
}

template<>
__forceinline__ __device__ char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, c1);
}

template<>
__forceinline__ __device__ char2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, c2);
}

template<>
__forceinline__ __device__ char4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, c4);
}

template<>
__forceinline__ __device__ short surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, s1);
}

template<>
__forceinline__ __device__ short2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, s2);
}

template<>
__forceinline__ __device__ short4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, s4);
}

template<>
__forceinline__ __device__ int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, u1);
}

template<>
__forceinline__ __device__ int2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, u2);
}

template<>
__forceinline__ __device__ int4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, u4);
}

template<>
__forceinline__ __device__ long long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern __device__ __device_builtin__ uchar1     __surf2Dreadc1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar2     __surf2Dreadc2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar4     __surf2Dreadc4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort1    __surf2Dreads1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort2    __surf2Dreads2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort4    __surf2Dreads4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint1      __surf2Dreadu1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint2      __surf2Dreadu2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint4      __surf2Dreadu4(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong1 __surf2Dreadl1(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong2 __surf2Dreadl2(surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, y, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2Dread##type(surf, x, y, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2Dread##type(surf, x, y, cudaBoundaryModeClamp) : \
                                           __surf2Dread##type(surf, x, y, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, y, mode, type) \
        __surf2Dread##type(surf, x, y, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surf2Dread(&tmp, surf, x, y, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surf2Dread<T>(surf, x, y, mode);
}

template<>
__forceinline__ __device__ char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, y, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, y, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, y, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, c1);
}

template<>
__forceinline__ __device__ char2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, y, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, c2);
}

template<>
__forceinline__ __device__ char4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, y, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, c4);
}

template<>
__forceinline__ __device__ short surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, y, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, y, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, s1);
}

template<>
__forceinline__ __device__ short2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, y, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, s2);
}

template<>
__forceinline__ __device__ short4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, y, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, s4);
}

template<>
__forceinline__ __device__ int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, y, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, y, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, u1);
}

template<>
__forceinline__ __device__ int2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, u2);
}

template<>
__forceinline__ __device__ int4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, u4);
}

template<>
__forceinline__ __device__ long long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, y, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, y, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, y, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, y, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, y, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, y, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, y, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, y, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, y, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern __device__ __device_builtin__ uchar1     __surf3Dreadc1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar2     __surf3Dreadc2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar4     __surf3Dreadc4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort1    __surf3Dreads1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort2    __surf3Dreads2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort4    __surf3Dreads4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint1      __surf3Dreadu1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint2      __surf3Dreadu2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint4      __surf3Dreadu4(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong1 __surf3Dreadl1(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong2 __surf3Dreadl2(surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, y, z, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf3Dread##type(surf, x, y, z, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf3Dread##type(surf, x, y, z, cudaBoundaryModeClamp) : \
                                           __surf3Dread##type(surf, x, y, z, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, y, z, mode, type) \
         __surf3Dread##type(surf, x, y, z, cudaBoundaryModeTrap)

#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, z, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, z, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, z, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, z, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, z, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surf3Dread(&tmp, surf, x, y, z, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surf3Dread<T>(surf, x, y, z, mode);
}

template<>
__forceinline__ __device__ char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, y, z, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, y, z, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, y, z, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, c1);
}

template<>
__forceinline__ __device__ char2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, y, z, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, c2);
}

template<>
__forceinline__ __device__ char4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, y, z, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, c4);
}

template<>
__forceinline__ __device__ short surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, y, z, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, y, z, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, s1);
}

template<>
__forceinline__ __device__ short2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, y, z, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, s2);
}

template<>
__forceinline__ __device__ short4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, y, z, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, s4);
}

template<>
__forceinline__ __device__ int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, y, z, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, y, z, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, u1);
}

template<>
__forceinline__ __device__ int2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, z, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, u2);
}

template<>
__forceinline__ __device__ int4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, z, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, u4);
}

template<>
__forceinline__ __device__ long long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, y, z, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, y, z, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, y, z, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, z, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, y, z, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, y, z, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, y, z, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, y, z, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, z, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, z, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, z, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, z, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, y, z, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, y, z, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, z, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, z, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern __device__ __device_builtin__ uchar1     __surf1DLayeredreadc1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar2     __surf1DLayeredreadc2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uchar4     __surf1DLayeredreadc4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort1    __surf1DLayeredreads1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort2    __surf1DLayeredreads2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ushort4    __surf1DLayeredreads4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint1      __surf1DLayeredreadu1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint2      __surf1DLayeredreadu2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ uint4      __surf1DLayeredreadu4(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong1 __surf1DLayeredreadl1(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ ulonglong2 __surf1DLayeredreadl2(surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, layer, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeClamp) : \
                                           __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, layer, mode, type) \
         __surf1DLayeredread##type(surf, x, layer, cudaBoundaryModeTrap)

#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, layer, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, layer, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, layer, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, layer, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, layer, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surf1DLayeredread(&tmp, surf, x, layer, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surf1DLayeredread<T>(surf, x, layer, mode);
}

template<>
__forceinline__ __device__ char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, layer, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, c1);
}

template<>
__forceinline__ __device__ char2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, layer, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, c2);
}

template<>
__forceinline__ __device__ char4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, layer, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, c4);
}

template<>
__forceinline__ __device__ short surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, layer, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, layer, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, s1);
}

template<>
__forceinline__ __device__ short2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, layer, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, s2);
}

template<>
__forceinline__ __device__ short4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, layer, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, s4);
}

template<>
__forceinline__ __device__ int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, u1);
}

template<>
__forceinline__ __device__ int2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, layer, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, u2);
}

template<>
__forceinline__ __device__ int4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, layer, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, u4);
}

template<>
__forceinline__ __device__ long long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, layer, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, layer, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, layer, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, layer, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, layer, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, layer, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, layer, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, layer, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, layer, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, layer, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, layer, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
template <typename T>
extern __device__ __device_builtin__ uchar1     __surf2DLayeredreadc1(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ uchar2     __surf2DLayeredreadc2(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ uchar4     __surf2DLayeredreadc4(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ ushort1    __surf2DLayeredreads1(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ ushort2    __surf2DLayeredreads2(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ ushort4    __surf2DLayeredreads4(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ uint1      __surf2DLayeredreadu1(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ uint2      __surf2DLayeredreadu2(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ uint4      __surf2DLayeredreadu4(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ ulonglong1 __surf2DLayeredreadl1(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ ulonglong2 __surf2DLayeredreadl2(T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, y, layer, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, y, layer, mode, type) \
         __surf2DLayeredread##type(surf, x, y, layer, cudaBoundaryModeTrap)

#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, layer, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, layer, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, layer, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, layer, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, layer, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surf2DLayeredread(&tmp, surf, x, y, layer, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surf2DLayeredread<T>(surf, x, y, layer, mode);
}

template<>
__forceinline__ __device__ char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, y, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, y, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, y, layer, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, c1);
}

template<>
__forceinline__ __device__ char2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, y, layer, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, c2);
}

template<>
__forceinline__ __device__ char4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, y, layer, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, c4);
}

template<>
__forceinline__ __device__ short surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, y, layer, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, y, layer, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, s1);
}

template<>
__forceinline__ __device__ short2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, y, layer, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, s2);
}

template<>
__forceinline__ __device__ short4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, y, layer, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, s4);
}

template<>
__forceinline__ __device__ int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, y, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, y, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, u1);
}

template<>
__forceinline__ __device__ int2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layer, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, u2);
}

template<>
__forceinline__ __device__ int4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layer, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, u4);
}

template<>
__forceinline__ __device__ long long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, y, layer, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, y, layer, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, y, layer, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layer, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, y, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, y, layer, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, y, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, y, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layer, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layer, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layer, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layer, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, y, layer, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, y, layer, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layer, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layer, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics 
#define __surfModeSwitch(surf, x, y, face, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */
// Cubemap and cubemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(surf, x, y, face, mode, type) \
         __surf2DLayeredread##type(surf, x, y, face, cudaBoundaryModeTrap)

#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, face, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, face, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, face, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, face, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, face, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surfCubemapread(&tmp, surf, x, y, face, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surfCubemapread<T>(surf, x, y, face, mode);
}

template<>
__forceinline__ __device__ char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, y, face, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, y, face, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, y, face, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, c1);
}

template<>
__forceinline__ __device__ char2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, y, face, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, c2);
}

template<>
__forceinline__ __device__ char4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, y, face, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, c4);
}

template<>
__forceinline__ __device__ short surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, y, face, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, y, face, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, s1);
}

template<>
__forceinline__ __device__ short2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, y, face, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, s2);
}

template<>
__forceinline__ __device__ short4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, y, face, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, s4);
}

template<>
__forceinline__ __device__ int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, y, face, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, y, face, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, u1);
}

template<>
__forceinline__ __device__ int2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, face, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, u2);
}

template<>
__forceinline__ __device__ int4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, face, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, u4);
}

template<>
__forceinline__ __device__ long long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, y, face, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, y, face, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, y, face, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, face, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, y, face, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, y, face, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, y, face, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, y, face, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, face, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, face, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, face, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, face, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, y, face, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, y, face, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, face, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, face, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(surf, x, y, layerFace, mode, type)                                                   \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeTrap ))

#else /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(surf, x, y, layerFace, mode, type) \
         __surf2DLayeredread##type(surf, x, y, layerFace, cudaBoundaryModeTrap)


#endif /* CUDA_ARCH && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  (s ==  1) ? (void)(*(uchar1 *)res = __surfModeSwitch(surf, x, y, layerFace, mode, c1)) :
  (s ==  2) ? (void)(*(ushort1*)res = __surfModeSwitch(surf, x, y, layerFace, mode, s1)) :
  (s ==  4) ? (void)(*(uint1  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u1)) :
  (s ==  8) ? (void)(*(uint2  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u2)) :
  (s == 16) ? (void)(*(uint4  *)res = __surfModeSwitch(surf, x, y, layerFace, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ T surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  T tmp;
  
  surfCubemapLayeredread(&tmp, surf, x, y, layerFace, (int)sizeof(T), mode);
  
  return tmp;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  *res = surfCubemapLayeredread<T>(surf, x, y, layerFace, mode);
}

template<>
__forceinline__ __device__ char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (char)__surfModeSwitch(surf, x, y, layerFace, mode, c1).x;
}

template<>
__forceinline__ __device__ signed char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (signed char)__surfModeSwitch(surf, x, y, layerFace, mode, c1).x;
}

template<>
__forceinline__ __device__ unsigned char surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, c1).x;
}

template<>
__forceinline__ __device__ char1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_char1((signed char)__surfModeSwitch(surf, x, y, layerFace, mode, c1).x);
}

template<>
__forceinline__ __device__ uchar1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, c1);
}

template<>
__forceinline__ __device__ char2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uchar2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, c2);
  
  return make_char2((signed char)tmp.x, (signed char)tmp.y);
}

template<>
__forceinline__ __device__ uchar2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, c2);
}

template<>
__forceinline__ __device__ char4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uchar4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, c4);
  
  return make_char4((signed char)tmp.x, (signed char)tmp.y, (signed char)tmp.z, (signed char)tmp.w);
}

template<>
__forceinline__ __device__ uchar4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, c4);
}

template<>
__forceinline__ __device__ short surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (short)__surfModeSwitch(surf, x, y, layerFace, mode, s1).x;
}

template<>
__forceinline__ __device__ unsigned short surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, s1).x;
}

template<>
__forceinline__ __device__ short1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_short1((signed short)__surfModeSwitch(surf, x, y, layerFace, mode, s1).x);
}

template<>
__forceinline__ __device__ ushort1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, s1);
}

template<>
__forceinline__ __device__ short2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  ushort2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, s2);
  
  return make_short2((signed short)tmp.x, (signed short)tmp.y);
}

template<>
__forceinline__ __device__ ushort2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, s2);
}

template<>
__forceinline__ __device__ short4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  ushort4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, s4);
  
  return make_short4((signed short)tmp.x, (signed short)tmp.y, (signed short)tmp.z, (signed short)tmp.w);
}

template<>
__forceinline__ __device__ ushort4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, s4);
}

template<>
__forceinline__ __device__ int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, u1).x;
}

template<>
__forceinline__ __device__ int1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_int1((signed int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x);
}

template<>
__forceinline__ __device__ uint1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, u1);
}

template<>
__forceinline__ __device__ int2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u2);
  
  return make_int2((int)tmp.x, (int)tmp.y);
}

template<>
__forceinline__ __device__ uint2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, u2);
}

template<>
__forceinline__ __device__ int4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u4);
  
  return make_int4((int)tmp.x, (int)tmp.y, (int)tmp.z, (int)tmp.w);
}

template<>
__forceinline__ __device__ uint4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, u4);
}

template<>
__forceinline__ __device__ long long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (long long int)__surfModeSwitch(surf, x, y, layerFace, mode, l1).x;
}

template<>
__forceinline__ __device__ unsigned long long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, l1).x;
}

template<>
__forceinline__ __device__ longlong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_longlong1((long long int)__surfModeSwitch(surf, x, y, layerFace, mode, l1).x);
}

template<>
__forceinline__ __device__ ulonglong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, l1);
}

template<>
__forceinline__ __device__ longlong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  ulonglong2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, l2);
  
  return make_longlong2((long long int)tmp.x, (long long int)tmp.y);
}

template<>
__forceinline__ __device__ ulonglong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __surfModeSwitch(surf, x, y, layerFace, mode, l2);
}

#if !defined(__LP64__)

template<>
__forceinline__ __device__ long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (long int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x;
}

template<>
__forceinline__ __device__ unsigned long int surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return (unsigned long int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x;
}

template<>
__forceinline__ __device__ long1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_long1((long int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x);
}

template<>
__forceinline__ __device__ ulong1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_ulong1((unsigned long int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x);
}

template<>
__forceinline__ __device__ long2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u2);
  
  return make_long2((long int)tmp.x, (long int)tmp.y);
}

template<>
__forceinline__ __device__ ulong2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u2);
  
  return make_ulong2((unsigned long int)tmp.x, (unsigned long int)tmp.y);
}

template<>
__forceinline__ __device__ long4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u4);
  
  return make_long4((long int)tmp.x, (long int)tmp.y, (long int)tmp.z, (long int)tmp.w);
}

template<>
__forceinline__ __device__ ulong4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u4);
  
  return make_ulong4((unsigned long int)tmp.x, (unsigned long int)tmp.y, (unsigned long int)tmp.z, (unsigned long int)tmp.w);
}

#endif /* !__LP64__ */

template<>
__forceinline__ __device__ float surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return __int_as_float((int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x);
}

template<>
__forceinline__ __device__ float1 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  return make_float1(__int_as_float((int)__surfModeSwitch(surf, x, y, layerFace, mode, u1).x));
}

template<>
__forceinline__ __device__ float2 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint2 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u2);
  
  return make_float2(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y));
}

template<>
__forceinline__ __device__ float4 surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode)
{
  uint4 tmp = __surfModeSwitch(surf, x, y, layerFace, mode, u4);
  
  return make_float4(__int_as_float((int)tmp.x), __int_as_float((int)tmp.y), __int_as_float((int)tmp.z), __int_as_float((int)tmp.w));
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ __device_builtin__ void __surf1Dwritec1(    uchar1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwritec2(    uchar2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwritec4(    uchar4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwrites1(   ushort1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwrites2(   ushort2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwrites4(   ushort4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwriteu1(     uint1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwriteu2(     uint2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwriteu4(     uint4 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType1D> t, int x, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1Dwrite##type(val, surf, x, cudaBoundaryModeClamp) : \
                                           __surf1Dwrite##type(val, surf, x, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, mode, type) \
        __surf1Dwrite##type(val, surf, x, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surf1Dwrite(val, surf, x, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(signed char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned char val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned short val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long int val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float1 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float2 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float4 val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ __device_builtin__ void __surf2Dwritec1(    uchar1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwritec2(    uchar2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwritec4(    uchar4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwrites1(   ushort1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwrites2(   ushort2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwrites4(   ushort4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwriteu1(     uint1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwriteu2(     uint2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwriteu4(     uint4 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf2Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType2D> t, int x, int y, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeClamp) : \
                                           __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, mode, type) \
        __surf2Dwrite##type(val, surf, x, y, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surf2Dwrite(val, surf, x, y, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(signed char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned char val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned short val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long int val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float1 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float2 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float4 val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ __device_builtin__ void __surf3Dwritec1(    uchar1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwritec2(    uchar2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwritec4(    uchar4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwrites1(   ushort1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwrites2(   ushort2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwrites4(   ushort4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwriteu1(     uint1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwriteu2(     uint2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwriteu4(     uint4 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwritel1(ulonglong1 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf3Dwritel2(ulonglong2 val, surface<void, cudaSurfaceType3D> t, int x, int y, int z, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, z, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeClamp) : \
                                           __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, z, mode, type) \
        __surf3Dwrite##type(val, surf, x, y, z, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, z, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, z, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, z, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, z, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, z, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surf3Dwrite(val, surf, x, y, z, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(signed char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned char val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned short val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long int val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float1 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float2 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float4 val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ __device_builtin__ void __surf1DLayeredwritec1(    uchar1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwritec2(    uchar2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwritec4(    uchar4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwrites1(   ushort1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwrites2(   ushort2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwrites4(   ushort4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwriteu1(     uint1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwriteu2(     uint2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwriteu4(     uint4 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwritel1(ulonglong1 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);
extern __device__ __device_builtin__ void __surf1DLayeredwritel2(ulonglong2 val, surface<void, cudaSurfaceType1DLayered> t, int x, int layer, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeClamp) : \
                                           __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, layer, mode, type) \
        __surf1DLayeredwrite##type(val, surf, x, layer, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, layer, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, layer, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, layer, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, layer, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, layer, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surf1DLayeredwrite(val, surf, x, layer, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(signed char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float1 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float2 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float4 val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec1(    uchar1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec2(    uchar2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritec4(    uchar4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites1(   ushort1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites2(   ushort2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwrites4(   ushort4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu1(     uint1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu2(     uint2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwriteu4(     uint4 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritel1(ulonglong1 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);
template <typename T>
extern __device__ __device_builtin__ void __surf2DLayeredwritel2(ulonglong2 val, T t, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, layer, mode, type)                                                    \
        ((mode == cudaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeZero ) : \
         (mode == cudaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeTrap ))

#else /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, layer, mode, type) \
        __surf2DLayeredwrite##type(val, surf, x, y, layer, cudaBoundaryModeTrap)

#endif /* __CUDA_ARCH__ && __CUDA_ARCH__ >= 200 */

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, layer, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, layer, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, layer, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, layer, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, layer, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surf2DLayeredwrite(val, surf, x, y, layer, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(signed char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned char val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned short val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long int val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float1 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float2 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float4 val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

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

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, face, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, face, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, face, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, face, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, face, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  surfCubemapwrite(val, surf, x, y, face, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(signed char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(char4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(short4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(int4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(uint4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(long4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float1 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float2 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapwrite(float4 val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

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

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
  union {
    T       val;
    uchar1  c1;
    ushort1 s1;
    uint1   u1;
    uint2   u2;
    uint4   u4;
  } tmp;
  
  tmp.val = val;
  
  (s ==  1) ? (void)(__surfModeSwitch(tmp.c1, surf, x, y, layerFace, mode, c1)) :
  (s ==  2) ? (void)(__surfModeSwitch(tmp.s1, surf, x, y, layerFace, mode, s1)) :
  (s ==  4) ? (void)(__surfModeSwitch(tmp.u1, surf, x, y, layerFace, mode, u1)) :
  (s ==  8) ? (void)(__surfModeSwitch(tmp.u2, surf, x, y, layerFace, mode, u2)) :
  (s == 16) ? (void)(__surfModeSwitch(tmp.u4, surf, x, y, layerFace, mode, u4)) :
              (void)0;
}

template<class T>
__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{;
  surfCubemapLayeredwrite(val, surf, x, y, layerFace, (int)sizeof(T), mode);
}


__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(signed char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned char val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(char4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uchar4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned short val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(short4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ushort4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(int4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(uint4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(longlong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulonglong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(unsigned long int val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(long4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(ulong4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float1 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float2 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

__SURFACE_FUNCTIONS_DECL__ void surfCubemapLayeredwrite(float4 val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap);

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#elif defined(__CUDABE__)

extern uchar1     __surf1Dreadc1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1Dreadc2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1Dreadc4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1Dreads1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1Dreads2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1Dreads4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1Dreadu1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1Dreadu2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1Dreadu4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1Dreadl1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1Dreadl2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2Dreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2Dreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2Dreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2Dreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2Dreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2Dreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2Dreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2Dreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2Dreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2Dreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2Dreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf3Dreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf3Dreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf3Dreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf3Dreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf3Dreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf3Dreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf3Dreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf3Dreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf3Dreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf3Dreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf3Dreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf1DLayeredreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1DLayeredreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1DLayeredreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1DLayeredreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1DLayeredreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1DLayeredreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1DLayeredreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1DLayeredreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1DLayeredreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1DLayeredreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1DLayeredreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2DLayeredreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2DLayeredreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2DLayeredreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2DLayeredreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2DLayeredreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2DLayeredreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2DLayeredreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2DLayeredreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2DLayeredreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2DLayeredreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2DLayeredreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec1(    uchar1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec2(    uchar2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec4(    uchar4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites1(   ushort1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites2(   ushort2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites4(   ushort4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu1(     uint1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu2(     uint2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu4(     uint4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel1(ulonglong1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel2(ulonglong2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec1(    uchar1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec2(    uchar2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec4(    uchar4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites1(   ushort1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites2(   ushort2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites4(   ushort4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu1(     uint1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu2(     uint2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu4(     uint4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel1(ulonglong1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel2(ulonglong2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec1(    uchar1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec2(    uchar2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec4(    uchar4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites1(   ushort1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites2(   ushort2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites4(   ushort4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu1(     uint1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu2(     uint2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu4(     uint4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
#endif /* __cplusplus && __CUDACC__ */

#undef __SURFACE_FUNCTIONS_DECL__

#if !defined(__CUDACC_RTC__)
#include "surface_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__SURFACE_FUNCTIONS_H__ */
