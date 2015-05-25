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

#if !defined(__DEVICE_FUNCTIONS_HPP__)
#define __DEVICE_FUNCTIONS_HPP__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__) || defined(__cplusplus) && defined(__CUDACC__)

#if defined(__CUDACC_RTC__)
#define __DEVICE_FUNCTIONS_DECL__ __host__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ int mulhi(int a, int b)
{
  return __mulhi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(int a, unsigned int b)
{
  return __umulhi((unsigned int)a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(unsigned int a, int b)
{
  return __umulhi(a, (unsigned int)b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long int mul64hi(long long int a, long long int b)
{
  return __mul64hi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b)
{
  return __umul64hi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(long long int a, unsigned long long int b)
{
  return __umul64hi((unsigned long long int)a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(unsigned long long int a, long long int b)
{
  return __umul64hi(a, (unsigned long long int)b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int float_as_int(float a)
{
  return __float_as_int(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float int_as_float(int a)
{
  return __int_as_float(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float saturate(float a)
{
  return __saturatef(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int mul24(int a, int b)
{
  return __mul24(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umul24(unsigned int a, unsigned int b)
{
  return __umul24(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void trap(void)
{
  __trap();
}

/* argument is optional, value of 0 means no value */
__DEVICE_FUNCTIONS_STATIC_DECL__ void brkpt(int c)
{
  __brkpt(c);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void syncthreads(void)
{
  __syncthreads();
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void prof_trigger(int e)
{
       if (e ==  0) __prof_trigger( 0);
  else if (e ==  1) __prof_trigger( 1);
  else if (e ==  2) __prof_trigger( 2);
  else if (e ==  3) __prof_trigger( 3);
  else if (e ==  4) __prof_trigger( 4);
  else if (e ==  5) __prof_trigger( 5);
  else if (e ==  6) __prof_trigger( 6);
  else if (e ==  7) __prof_trigger( 7);
  else if (e ==  8) __prof_trigger( 8);
  else if (e ==  9) __prof_trigger( 9);
  else if (e == 10) __prof_trigger(10);
  else if (e == 11) __prof_trigger(11);
  else if (e == 12) __prof_trigger(12);
  else if (e == 13) __prof_trigger(13);
  else if (e == 14) __prof_trigger(14);
  else if (e == 15) __prof_trigger(15);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void threadfence(bool global)
{
  global ? __threadfence() : __threadfence_block();
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int float2int(float a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __float2int_rn(a) :
         mode == cudaRoundPosInf  ? __float2int_ru(a) :
         mode == cudaRoundMinInf  ? __float2int_rd(a) :
                                    __float2int_rz(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int float2uint(float a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __float2uint_rn(a) :
         mode == cudaRoundPosInf  ? __float2uint_ru(a) :
         mode == cudaRoundMinInf  ? __float2uint_rd(a) :
                                    __float2uint_rz(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float int2float(int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __int2float_rz(a) :
         mode == cudaRoundPosInf ? __int2float_ru(a) :
         mode == cudaRoundMinInf ? __int2float_rd(a) :
                                   __int2float_rn(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float uint2float(unsigned int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __uint2float_rz(a) :
         mode == cudaRoundPosInf ? __uint2float_ru(a) :
         mode == cudaRoundMinInf ? __uint2float_rd(a) :
                                   __uint2float_rn(a);
}

#undef __DEVICE_FUNCTIONS_DECL__
#undef __DEVICE_FUNCTIONS_STATIC_DECL__

#endif /* __CUDACC_RTC__ || __cplusplus && __CUDACC__ */
#if defined(__CUDACC_RTC__) || defined(__CUDABE__)

#if defined(__CUDACC_RTC__)
#define __DEVICE_FUNCTIONS_DECL__ __host__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ static __forceinline__ 
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/*******************************************************************************
*                                                                              *
* SYNCHRONIZATION FUNCTIONS                                                    *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_count(int predicate)
{
  return __nvvm_bar0_popc(predicate);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_and(int predicate)
{
  return __nvvm_bar0_and(predicate);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_or(int predicate)
{
  return __nvvm_bar0_or(predicate);
}

/*******************************************************************************
*                                                                              *
* MEMORY FENCE FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence_block()
{
  __nvvm_membar_cta();
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence()
{
  __nvvm_membar_gl();
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence_system()
{
  __nvvm_membar_sys();
}

/*******************************************************************************
*                                                                              *
* VOTE FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int __all(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        ".reg .pred \t%%p2; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.all.pred \t%%p2, %%p1; \n\t"
        "selp.s32 \t%0, 1, 0, %%p2; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __any(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        ".reg .pred \t%%p2; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.any.pred \t%%p2, %%p1; \n\t"
        "selp.s32 \t%0, 1, 0, %%p2; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}

__DEVICE_FUNCTIONS_STATIC_DECL__
#if defined(__CUDACC_RTC__)
unsigned int
#else /* !__CUDACC_RTC__ */
int
#endif /* __CUDACC_RTC__ */
__ballot(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.ballot.b32 \t%0, %%p1; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}

/*******************************************************************************
*                                                                              *
* MISCELLANEOUS FUNCTIONS                                                      *
*                                                                              *
*******************************************************************************/
#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ void __brkpt(int)
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ void __brkpt()
#endif /* __CUDACC_RTC__ */
{
  asm __volatile__ ("brkpt;");
}

#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ clock_t clock()
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int clock()
#endif /* __CUDACC_RTC__ */
{
  int r;
  asm __volatile__ ("mov.u32 \t%0, %%clock;" : "=r"(r));
  return r;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long clock64()
{
  long long z;
  asm __volatile__ ("mov.u64 \t%0, %%clock64;" : "=l"(z));
  return z;
}
    
#define __prof_trigger(X) asm __volatile__ ("pmevent \t" #X ";")

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm0(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm0;" : "=r"(r));
  return r;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm1(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm1;" : "=r"(r));
  return r;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm2(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm2;" : "=r"(r));
  return r;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm3(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm3;" : "=r"(r));
  return r;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void __trap(void)
{
  asm __volatile__ ("trap;");
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void* memcpy(void *dest, const void *src, size_t n)
{
  __nvvm_memcpy((unsigned char *)dest, (unsigned char *)src, n, 
                /*alignment=*/ 1);
  return dest;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void* memset(void *dest, int c, size_t n)
{
  __nvvm_memset((unsigned char *)dest, (unsigned char)c, n, 
                /*alignment=*/1);
  return dest;
}

/*******************************************************************************
*                                                                              *
* MATH FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int __clz(int x)
{
  return __nv_clz(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __clzll(long long x)
{
  return __nv_clzll(x);
}

#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popc(unsigned int x)
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popc(int x)
#endif /* __CUDACC_RTC__ */
{
  return __nv_popc(x);
}

#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popcll(unsigned long long x)
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popcll(long long x)
#endif /* __CUDACC_RTC__ */
{
  return __nv_popcll(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __byte_perm(unsigned int a,
                                                unsigned int b,
                                                unsigned int c)
{
  return __nv_byte_perm(a, b, c);
}

/*******************************************************************************
*                                                                              *
* INTEGER MATH FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int min(int x, int y)
{
  return __nv_min(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umin(unsigned int x, unsigned int y)
{
  return __nv_umin(x, y);
}
    
__DEVICE_FUNCTIONS_STATIC_DECL__ long long llmin(long long x, long long y)
{
  return __nv_llmin(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long ullmin(unsigned long long x,
                                                 unsigned long long y)
{
  return __nv_ullmin(x, y);
}
    
__DEVICE_FUNCTIONS_STATIC_DECL__ int max(int x, int y)
{
  return __nv_max(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umax(unsigned int x, unsigned int y)
{
  return __nv_umax(x, y);
}
    
__DEVICE_FUNCTIONS_STATIC_DECL__ long long llmax(long long x, long long y)
{
  return __nv_llmax(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long ullmax(unsigned long long x,
                                                 unsigned long long y)
{
  return __nv_ullmax(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __mulhi(int x, int y)
{
  return __nv_mulhi(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __umulhi(unsigned int x, unsigned int y)
{
  return __nv_umulhi(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __mul64hi(long long x, long long y)
{
  return __nv_mul64hi(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __umul64hi(unsigned long long x,
                                                     unsigned long long y)
{
  return __nv_umul64hi(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __mul24(int x, int y)
{
  return __nv_mul24(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __umul24(unsigned int x, unsigned int y)
{
  return __nv_umul24(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __brev(unsigned int x)
{
  return __nv_brev(x);
}
    
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __brevll(unsigned long long x)
{
  return __nv_brevll(x);
}
    
#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __sad(int x, int y, unsigned int z)
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __sad(int x, int y, int z)
#endif /* __CUDACC_RTC__ */
{
  return __nv_sad(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __usad(unsigned int x,
                                           unsigned int y,
                                           unsigned int z)
{
  return __nv_usad(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int abs(int x)
{
  return __nv_abs(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long labs(long x)
{
#if defined(__LP64__)
  return __nv_llabs((long long) x);
#else /* __LP64__ */
  return __nv_abs((int) x);
#endif /* __LP64__ */
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long llabs(long long x)
{
  return __nv_llabs(x);
}

/*******************************************************************************
*                                                                              *
* FP MATH FUNCTIONS                                                            *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float floorf(float f)
{
  return __nv_floorf(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double floor(double f)
{
  return __nv_floor(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float fabsf(float f)
{
  return __nv_fabsf(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double fabs(double f)
{
  return __nv_fabs(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __rcp64h(double d)
{
  return __nv_rcp64h(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y)
{
  return __nv_fminf(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float fmaxf(float x, float y)
{
  return __nv_fmaxf(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float rsqrtf(float x)
{
  return __nv_rsqrtf(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double fmin(double x, double y)
{
  return __nv_fmin(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double fmax(double x, double y)
{
  return __nv_fmax(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double rsqrt(double x)
{
  return __nv_rsqrt(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double ceil(double x)
{
  return __nv_ceil(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double trunc(double x)
{
  return __nv_trunc(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float exp2f(float x)
{
  return __nv_exp2f(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float truncf(float x)
{
  return __nv_truncf(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float ceilf(float x)
{
  return __nv_ceilf(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __saturatef(float x)
{
  return __nv_saturatef(x);
}

/*******************************************************************************
*                                                                              *
* FMAF                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rn(float x, float y, float z)
{
  return __nv_fmaf_rn(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rz(float x, float y, float z)
{
  return __nv_fmaf_rz(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rd(float x, float y, float z)
{
  return __nv_fmaf_rd(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ru(float x, float y, float z)
{
  return __nv_fmaf_ru(x, y, z);
}

/*******************************************************************************
*                                                                              *
* FMAF_IEEE                                                                    *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rn(float x, float y, float z)
{
  return __nv_fmaf_ieee_rn(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rz(float x, float y, float z)
{
  return __nv_fmaf_ieee_rz(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rd(float x, float y, float z)
{
  return __nv_fmaf_ieee_rd(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_ru(float x, float y, float z)
{
  return __nv_fmaf_ieee_ru(x, y, z);
}

/*******************************************************************************
*                                                                              *
* FMA                                                                          *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rn(double x, double y, double z)
{
  return __nv_fma_rn(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rz(double x, double y, double z)
{
  return __nv_fma_rz(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rd(double x, double y, double z)
{
  return __nv_fma_rd(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_ru(double x, double y, double z)
{
  return __nv_fma_ru(x, y, z);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdividef(float x, float y)
{
  return __nv_fast_fdividef(x, y);
}

/*******************************************************************************
*                                                                              *
* FDIV                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rn(float x, float y)
{
  return __nv_fdiv_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rz(float x, float y)
{
  return __nv_fdiv_rz(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rd(float x, float y)
{
  return __nv_fdiv_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_ru(float x, float y)
{
  return __nv_fdiv_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* FRCP                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rn(float x)
{
  return __nv_frcp_rn(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rz(float x)
{
  return __nv_frcp_rz(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rd(float x)
{
  return __nv_frcp_rd(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_ru(float x)
{
  return __nv_frcp_ru(x);
}

/*******************************************************************************
*                                                                              *
* FSQRT                                                                        *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rn(float x)
{
  return __nv_fsqrt_rn(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rz(float x)
{
  return __nv_fsqrt_rz(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rd(float x)
{
  return __nv_fsqrt_rd(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_ru(float x)
{
  return __nv_fsqrt_ru(x);
}

/*******************************************************************************
*                                                                              *
* DDIV                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rn(double x, double y)
{
  return __nv_ddiv_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rz(double x, double y)
{
  return __nv_ddiv_rz(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rd(double x, double y)
{
  return __nv_ddiv_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_ru(double x, double y)
{
  return __nv_ddiv_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* DRCP                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rn(double x)
{
  return __nv_drcp_rn(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rz(double x)
{
  return __nv_drcp_rz(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rd(double x)
{
  return __nv_drcp_rd(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_ru(double x)
{
  return __nv_drcp_ru(x);
}

/*******************************************************************************
*                                                                              *
* DSQRT                                                                        *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rn(double x)
{
  return __nv_dsqrt_rn(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rz(double x)
{
  return __nv_dsqrt_rz(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rd(double x)
{
  return __nv_dsqrt_rd(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_ru(double x)
{
  return __nv_dsqrt_ru(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float sqrtf(float x)
{
  return __nv_sqrtf(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double sqrt(double x)
{
  return __nv_sqrt(x);
}

/*******************************************************************************
*                                                                              *
* DADD                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rn(double x, double y)
{
  return __nv_dadd_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rz(double x, double y)
{
  return __nv_dadd_rz(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rd(double x, double y)
{
  return __nv_dadd_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_ru(double x, double y)
{
  return __nv_dadd_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* DMUL                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rn(double x, double y)
{
  return __nv_dmul_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rz(double x, double y)
{
  return __nv_dmul_rz(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rd(double x, double y)
{
  return __nv_dmul_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_ru(double x, double y)
{
  return __nv_dmul_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* FADD                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rd(float x, float y)
{
  return __nv_fadd_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_ru(float x, float y)
{
  return __nv_fadd_ru(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rn(float x, float y)
{
  return __nv_fadd_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rz(float x, float y)
{
  return __nv_fadd_rz(x, y);
}

/*******************************************************************************
*                                                                              *
* FMUL                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rd(float x, float y)
{
  return __nv_fmul_rd(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_ru(float x, float y)
{
  return __nv_fmul_ru(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rn(float x, float y)
{
  return __nv_fmul_rn(x, y);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rz(float x, float y)
{
  return __nv_fmul_rz(x, y);
}

/*******************************************************************************
*                                                                              *
* CONVERSION FUNCTIONS                                                         *
*                                                                              *
*******************************************************************************/
/* double to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rn(double d)
{
  return __nv_double2float_rn(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rz(double d)
{
  return __nv_double2float_rz(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rd(double d)
{
  return __nv_double2float_rd(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_ru(double d)
{
  return __nv_double2float_ru(d);
}
    
/* double to int */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rn(double d)
{
  return __nv_double2int_rn(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rz(double d)
{
  return __nv_double2int_rz(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rd(double d)
{
  return __nv_double2int_rd(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_ru(double d)
{
  return __nv_double2int_ru(d);
}

/* double to uint */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rn(double d)
{
  return __nv_double2uint_rn(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rz(double d)
{
  return __nv_double2uint_rz(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rd(double d)
{
  return __nv_double2uint_rd(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_ru(double d)
{
  return __nv_double2uint_ru(d);
}

/* int to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __int2double_rn(int i)
{
  return __nv_int2double_rn(i);
}

/* uint to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __uint2double_rn(unsigned int i)
{
  return __nv_uint2double_rn(i);
}

/* float to int */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rn(float in)
{
  return __nv_float2int_rn(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rz(float in)
{
  return __nv_float2int_rz(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rd(float in)
{
  return __nv_float2int_rd(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_ru(float in)
{
  return __nv_float2int_ru(in);
}

/* float to uint */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rn(float in)
{
  return __nv_float2uint_rn(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rz(float in)
{
  return __nv_float2uint_rz(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rd(float in)
{
  return __nv_float2uint_rd(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_ru(float in)
{
  return __nv_float2uint_ru(in);
}

/* int to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rn(int in)
{
  return __nv_int2float_rn(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rz(int in)
{
  return __nv_int2float_rz(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rd(int in)
{
  return __nv_int2float_rd(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_ru(int in)
{
  return __nv_int2float_ru(in);
}

/* unsigned int to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rn(unsigned int in)
{
  return __nv_uint2float_rn(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rz(unsigned int in)
{
  return __nv_uint2float_rz(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rd(unsigned int in)
{
  return __nv_uint2float_rd(in);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_ru(unsigned int in)
{
  return __nv_uint2float_ru(in);
}

/* hiloint vs double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __hiloint2double(int a, int b)
{
  return __nv_hiloint2double(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2loint(double d)
{
  return __nv_double2loint(d);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2hiint(double d)
{
  return __nv_double2hiint(d);
}

/* float to long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rn(float f)
{
  return __nv_float2ll_rn(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rz(float f)
{
  return __nv_float2ll_rz(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rd(float f)
{
  return __nv_float2ll_rd(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_ru(float f)
{
  return __nv_float2ll_ru(f);
}

/* float to unsigned long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rn(float f)
{
  return __nv_float2ull_rn(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rz(float f)
{
  return __nv_float2ull_rz(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rd(float f)
{
  return __nv_float2ull_rd(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_ru(float f)
{
  return __nv_float2ull_ru(f);
}

/* double to long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rn(double f)
{
  return __nv_double2ll_rn(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rz(double f)
{
  return __nv_double2ll_rz(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rd(double f)
{
  return __nv_double2ll_rd(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_ru(double f)
{
  return __nv_double2ll_ru(f);
}

/* double to unsigned long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rn(double f)
{
  return __nv_double2ull_rn(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rz(double f)
{
  return __nv_double2ull_rz(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rd(double f)
{
  return __nv_double2ull_rd(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_ru(double f)
{
  return __nv_double2ull_ru(f);
}

/* long long to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rn(long long l)
{
  return __nv_ll2float_rn(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rz(long long l)
{
  return __nv_ll2float_rz(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rd(long long l)
{
  return __nv_ll2float_rd(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_ru(long long l)
{
  return __nv_ll2float_ru(l);
}

/* unsigned long long to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rn(unsigned long long l)
{
  return __nv_ull2float_rn(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rz(unsigned long long l)
{
  return __nv_ull2float_rz(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rd(unsigned long long l)
{
  return __nv_ull2float_rd(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_ru(unsigned long long l)
{
  return __nv_ull2float_ru(l);
}

/* long long to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rn(long long l)
{
  return __nv_ll2double_rn(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rz(long long l)
{
  return __nv_ll2double_rz(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rd(long long l)
{
  return __nv_ll2double_rd(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_ru(long long l)
{
  return __nv_ll2double_ru(l);
}

/* unsigned long long to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rn(unsigned long long l)
{
  return __nv_ull2double_rn(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rz(unsigned long long l)
{
  return __nv_ull2double_rz(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rd(unsigned long long l)
{
  return __nv_ull2double_rd(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_ru(unsigned long long l)
{
  return __nv_ull2double_ru(l);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned short __float2half_rn(float f)
{
  return __nv_float2half_rn(f);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __half2float(unsigned short h)
{
  return __nv_half2float(h);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int_as_float(int x)
{
  return __nv_int_as_float(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float_as_int(float x)
{
  return __nv_float_as_int(x);
}
    
__DEVICE_FUNCTIONS_STATIC_DECL__ double __longlong_as_double(long long x)
{
  return __nv_longlong_as_double(x);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long  __double_as_longlong (double x)
{
  return __nv_double_as_longlong(x);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ float __sinf(float a)
{
  return __nv_fast_sinf(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __cosf(float a)
{
  return __nv_fast_cosf(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __log2f(float a)
{
  return __nv_fast_log2f(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ float __tanf(float a)
{
  return __nv_fast_tanf(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ void __sincosf(float a, float *sptr, float *cptr)
{
  __nv_fast_sincosf(a, sptr, cptr);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __expf(float a)
{
  return __nv_fast_expf(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __exp10f(float a)
{
  return __nv_fast_exp10f(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __log10f(float a)
{
  return __nv_fast_log10f(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __logf(float a)
{
  return __nv_fast_logf(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __powf(float a, float b)
{
  return __nv_fast_powf(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float fdividef(float a, float b)
{
#if defined(__USE_FAST_MATH__) && !defined(__CUDA_PREC_DIV)
  return __nv_fast_fdividef(a, b);
#else /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
  return a / b;
#endif /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
}

__DEVICE_FUNCTIONS_STATIC_DECL__ double fdivide(double a, double b)
{
  return a / b;
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __hadd(int a, int b)
{
  return __nv_hadd(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __rhadd(int a, int b)
{
  return __nv_rhadd(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __uhadd(unsigned int a, unsigned int b)
{
  return __nv_uhadd(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __urhadd(unsigned int a, unsigned int b)
{
  return __nv_urhadd(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rn (float a, float b)
{
  return __nv_fsub_rn(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rz (float a, float b)
{
  return __nv_fsub_rz(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rd (float a, float b)
{
  return __nv_fsub_rd(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_ru (float a, float b)
{
  return __nv_fsub_ru(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frsqrt_rn (float a)
{
  return __nv_frsqrt_rn(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __ffs(int a)
{
  return __nv_ffs(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int __ffsll(long long int a)
{
  return __nv_ffsll(a);
}

/*******************************************************************************
*                                                                              *
* ATOMIC OPERATIONS                                                            *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicAdd(int *p, int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicAdd(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, (int)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicAdd(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_add_gen_ll((volatile long long *)p, (long long)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
float __fAtomicAdd(float *p, float val)
{
  return __nvvm_atom_add_gen_f((volatile float *)p, val);
}


#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 600
__DEVICE_FUNCTIONS_STATIC_DECL__
double __dAtomicAdd(double *p, double val)
{
  return __nvvm_atom_add_gen_d((volatile double *)p, val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 600 */


__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicExch(int *p, int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicExch(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, (int)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicExch(unsigned long long *p,
                                   unsigned long long val)
{
  return __nvvm_atom_xchg_gen_ll((volatile long long *)p, (long long)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
float __fAtomicExch(float *p, float val)
{
  int old = __nvvm_atom_xchg_gen_i((volatile int *)p, __float_as_int(val));
  return __int_as_float(old);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicMin(int *p, int val)
{
  return __nvvm_atom_min_gen_i((volatile int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long __illAtomicMin(long long *p, long long val)
{
  return __nvvm_atom_min_gen_ll((volatile long long *)p, val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicMin(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_min_gen_ui((volatile unsigned int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicMin(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_min_gen_ull((volatile unsigned long long *)p, val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicMax(int *p, int val)
{
  return __nvvm_atom_max_gen_i((volatile int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long __illAtomicMax(long long *p, long long val)
{
  return __nvvm_atom_max_gen_ll((volatile long long *)p, val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicMax(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_max_gen_ui((unsigned int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicMax(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_max_gen_ull((volatile unsigned long long *)p, val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicInc(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_inc_gen_ui((unsigned int *)p, val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicDec(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_dec_gen_ui((unsigned int *)p, val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicCAS(int *p, int compare, int val)
{
  return __nvvm_atom_cas_gen_i((int *)p, compare, val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicCAS(unsigned int *p, unsigned int compare,
                          unsigned int val)
{
  return (unsigned int)__nvvm_atom_cas_gen_i((volatile int *)p,
                                             (int)compare,
                                             (int)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicCAS(unsigned long long int *p,
                                      unsigned long long int compare,
                                      unsigned long long int val)
{
  return
    (unsigned long long int)__nvvm_atom_cas_gen_ll((volatile long long int *)p,
                                                   (long long int)compare,
                                                   (long long int)val);
}

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicAnd(int *p, int val)
{
  return __nvvm_atom_and_gen_i((volatile int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicAnd(long long int *p, long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicAnd(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_and_gen_i((volatile int *)p, (int)val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicAnd(unsigned long long int *p,
                                      unsigned long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicOr(int *p, int val)
{
  return __nvvm_atom_or_gen_i((volatile int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicOr(long long int *p, long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicOr(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_or_gen_i((volatile int *)p, (int)val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicOr(unsigned long long int *p,
                                     unsigned long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicXor(int *p, int val)
{
  return __nvvm_atom_xor_gen_i((volatile int *)p, val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicXor(long long int *p, long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicXor(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_xor_gen_i((volatile int *)p, (int)val);
}

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicXor(unsigned long long int *p,
                                      unsigned long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

/*******************************************************************************
 *                                                                             *
 *                          SIMD functions                                     *
 *                                                                             *
 *******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabs2(unsigned int a)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int b = 0, c = 0;
    asm ("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                       \n\t"
         ".reg .u32 a,m,r;        \n\t"
         "mov.b32  a,%1;          \n\t"
         "prmt.b32 m,a,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
         "xor.b32  r,a,m;         \n\t" // conditionally invert bits
         "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
         "add.u32  r,r,m;         \n\t" // conditionally add 1
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise absolute value, with wrap-around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsss2(unsigned int a)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int b = 0, c = 0;
    asm("vabsdiff2.s32.s32.s32.sat %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                       \n\t"
         ".reg .u32 a,m,r;        \n\t"
         "mov.b32  a,%1;          \n\t"
         "prmt.b32 m,a,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
         "xor.b32  r,a,m;         \n\t" // conditionally invert bits
         "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
         "add.u32  r,r,m;         \n\t" // conditionally add 1
         "prmt.b32 m,r,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
         "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
         "sub.u32  r,r,m;         \n\t" // subtract 1 if result wrapped around
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise absolute value with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vadd2(unsigned int a, unsigned int b)
{
    unsigned int s, t;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vadd2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = a ^ b;          // sum bits
    t = a + b;          // actual sum
    s = s ^ t;          // determine carry-ins for each bit position
    s = s & 0x00010000; // carry-in to high word (= carry-out from low word)
    t = t - s;          // subtract out carry-out from low word
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return t;           // halfword-wise sum, with wrap around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddss2 (unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vadd2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    int ahi, alo, blo, bhi, rhi, rlo;
    ahi = (int)((a & 0xffff0000U));
    bhi = (int)((b & 0xffff0000U));
#if __CUDA_ARCH__ < 350
    alo = (int)(a << 16);
    blo = (int)(b << 16);
#else  /* __CUDA_ARCH__ < 350 */
    asm ("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(alo) : "r"(a));
    asm ("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(blo) : "r"(b));
#endif /* __CUDA_ARCH__ < 350 */
    asm ("add.sat.s32 %0,%1,%2;" : "=r"(rlo) : "r"(alo), "r"(blo));
    asm ("add.sat.s32 %0,%1,%2;" : "=r"(rhi) : "r"(ahi), "r"(bhi));
    asm ("prmt.b32 %0,%1,%2,0x7632;" : "=r"(r) : "r"(rlo), "r"(rhi));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // halfword-wise sum with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddus2 (unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vadd2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    int alo, blo, rlo, ahi, bhi, rhi;
    asm ("{                              \n\t"
         "and.b32     %0, %4, 0xffff;    \n\t"
         "and.b32     %1, %5, 0xffff;    \n\t"
#if __CUDA_ARCH__ < 350
         "shr.u32     %2, %4, 16;        \n\t"
         "shr.u32     %3, %5, 16;        \n\t"
#else  /* __CUDA_ARCH__ < 350 */
         "shf.r.clamp.b32  %2, %4, 0, 16;\n\t"
         "shf.r.clamp.b32  %3, %5, 0, 16;\n\t"
#endif /* __CUDA_ARCH__ < 350 */
         "}"
         : "=r"(alo), "=r"(blo), "=r"(ahi), "=r"(bhi) 
         : "r"(a), "r"(b));
    rlo = min (alo + blo, 65535);
    rhi = min (ahi + bhi, 65535);
    r = (rhi << 16) + rlo;
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise sum with unsigned saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgs2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vavrg2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The 
    // two expressions can be re-written as follows to avoid needing additional
    // intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
    // ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
    // between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
    // ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
    // means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
    // depending on the sign bit of the shifted sum. By handling the msb sum 
    // bit of the result separately, we avoid carry-out during summation and
    // also can use (potentially faster) logical right shifts.
    asm ("{                      \n\t"
         ".reg .u32 a,b,c,r,s,t,u,v;\n\t"
         "mov.b32 a,%1;          \n\t"
         "mov.b32 b,%2;          \n\t"
         "and.b32 u,a,0xfffefffe;\n\t" // prevent shift crossing chunk boundary
         "and.b32 v,b,0xfffefffe;\n\t" // prevent shift crossing chunk boundary
         "xor.b32 s,a,b;         \n\t" // a ^ b
         "and.b32 t,a,b;         \n\t" // a & b
         "shr.u32 u,u,1;         \n\t" // a >> 1
         "shr.u32 v,v,1;         \n\t" // b >> 1
         "and.b32 c,s,0x00010001;\n\t" // (a ^ b) & 1
         "and.b32 s,s,0x80008000;\n\t" // extract msb (a ^ b)
         "and.b32 t,t,0x00010001;\n\t" // (a & b) & 1
         "add.u32 r,u,v;         \n\t" // (a>>1)+(b>>1) 
         "add.u32 r,r,t;         \n\t" // (a>>1)+(b>>1)+(a&b&1); rec. msb cy-in
         "xor.b32 r,r,s;         \n\t" // compute msb sum bit: a ^ b ^ cy-in
         "shr.u32 t,r,15;        \n\t" // sign ((a + b) >> 1)
         "not.b32 t,t;           \n\t" // ~sign ((a + b) >> 1)
         "and.b32 t,t,c;         \n\t" // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
         "add.u32 r,r,t;         \n\t" // conditionally add ((a ^ b) & 1)
         "mov.b32 %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // halfword-wise average of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vavrg2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
    // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
    c = a ^ b;           
    r = a | b;
    c = c & 0xfffefffe; // ensure shift doesn't cross half-word boundaries
    c = c >> 1;
    r = r - c;
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise average of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vhaddu2(unsigned int a, unsigned int b)
{
    // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
    // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
    unsigned int r, s;
    s = a ^ b;
    r = a & b;
    s = s & 0xfffefffe; // ensure shift doesn't cross halfword boundaries
    s = s >> 1;
    r = r + s;
    return r;           // halfword-wise average of unsigned ints, rounded down
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpeq2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r & ~c;         // msb = 1, if r was 0x0000
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// convert msbs to mask
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise (un)signed eq comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpges2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.ge.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.ge.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed gt-eq comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vavgu2 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned gt-eq comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgts2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.gt.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.gt.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed gt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vhaddu2 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned gt comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmples2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.le.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.le.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed lt-eq comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpleu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vavgu2 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned lt-eq comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmplts2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.lt.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.lt.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed lt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpltu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vhaddu2 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned lt comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpne2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r | c;          // msb = 1, if r was not 0x0000
    asm ("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise (un)signed ne comparison, mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vabsdiff2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int t, u, v;
    s = a & 0x0000ffff; // extract low halfword
    r = b & 0x0000ffff; // extract low halfword
    u = umax (r, s);    // maximum of low halfwords
    v = umin (r, s);    // minimum of low halfwords
    s = a & 0xffff0000; // extract high halfword
    r = b & 0xffff0000; // extract high halfword
    t = umax (r, s);    // maximum of high halfwords
    s = umin (r, s);    // minimum of high halfwords
    r = u | t;          // maximum of both halfwords
    s = v | s;          // minimum of both halfwords
    r = r - s;          // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // halfword-wise absolute difference of unsigned ints
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxs2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmax2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int t, u;
    asm ("cvt.s32.s16 %0,%1;" : "=r"(r) : "r"(a)); // extract low halfword
    asm ("cvt.s32.s16 %0,%1;" : "=r"(s) : "r"(b)); // extract low halfword
    t = max((int)r,(int)s); // maximum of low halfwords
    r = a & 0xffff0000;     // extract high halfword
    s = b & 0xffff0000;     // extract high halfword
    u = max((int)r,(int)s); // maximum of high halfwords
    r = u | (t & 0xffff);   // combine halfword maximums
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise maximum of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmax2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int t, u;
    r = a & 0x0000ffff; // extract low halfword
    s = b & 0x0000ffff; // extract low halfword
    t = umax (r, s);    // maximum of low halfwords
    r = a & 0xffff0000; // extract high halfword
    s = b & 0xffff0000; // extract high halfword
    u = umax (r, s);    // maximum of high halfwords
    r = t | u;          // combine halfword maximums
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise maximum of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmins2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmin2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    unsigned int t, u;
    asm ("cvt.s32.s16 %0,%1;" : "=r"(r) : "r"(a)); // extract low halfword
    asm ("cvt.s32.s16 %0,%1;" : "=r"(s) : "r"(b)); // extract low halfword
    t = min((int)r,(int)s); // minimum of low halfwords
    r = a & 0xffff0000;     // extract high halfword
    s = b & 0xffff0000;     // extract high halfword
    u = min((int)r,(int)s); // minimum of high halfwords
    r = u | (t & 0xffff);   // combine halfword minimums
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise minimum of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vminu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmin2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int t, u;
    r = a & 0x0000ffff; // extract low halfword
    s = b & 0x0000ffff; // extract low halfword
    t = umin (r, s);    // minimum of low halfwords
    r = a & 0xffff0000; // extract high halfword
    s = b & 0xffff0000; // extract high halfword
    u = umin (r, s);    // minimum of high halfwords
    r = t | u;          // combine halfword minimums
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise minimum of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vseteq2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r & ~c;         // msb = 1, if r was 0x0000
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise (un)signed eq comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetges2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.ge.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.ge.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed gt-eq comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgeu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vavgu2 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned gt-eq comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgts2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.gt.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.gt.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed gt comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgtu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vhaddu2 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80008000; // msbs = carry-outs
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned gt comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetles2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.le.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.le.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed lt-eq comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetleu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vavgu2 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned lt-eq comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetlts2(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                             \n\t"
         ".reg .u32 a, b, r, s, t, u;   \n\t"
         "mov.b32        a,%1;          \n\t" 
         "mov.b32        b,%2;          \n\t"
         "and.b32        s,a,0xffff0000;\n\t" // high word of a
         "and.b32        t,b,0xffff0000;\n\t" // high word of b
         "set.lt.s32.s32 u,s,t;         \n\t" // compare two high words
         "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
         "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
         "set.lt.s32.s32 s,s,t;         \n\t" // compare two low words
         "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
         "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
         "mov.b32        %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise signed lt comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetltu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vhaddu2 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise unsigned lt comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetne2(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r | c;          // msb = 1, if r was not 0x0000
    c = c & 0x80008000; // extract msbs
    r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise (un)signed ne comparison, bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsadu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm("vabsdiff2.u32.u32.u32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int t, u, v;
    s = a & 0x0000ffff; // extract low halfword
    r = b & 0x0000ffff; // extract low halfword
    u = umax (r, s);    // maximum of low halfwords
    v = umin (r, s);    // minimum of low halfwords
    s = a & 0xffff0000; // extract high halfword
    r = b & 0xffff0000; // extract high halfword
    t = umax (r, s);    // maximum of high halfwords
    s = umin (r, s);    // minimum of high halfwords
    u = u - v;          // low halfword: |a - b| = max(a,b) - min(a,b); 
    t = t - s;          // high halfword: |a - b| = max(a,b) - min(a,b);
#if __CUDA_ARCH__ < 350
    asm ("shr.u32 %0,%0,16;" : "+r"(t));
#else  /*__CUDA_ARCH__ < 350 */
    asm ("shf.r.clamp.b32  %0,%0,0,16;" : "+r"(t));
#endif /*__CUDA_ARCH__ < 350 */
    r = t + u;          // sum absolute halfword differences
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise sum of abs differences of unsigned int
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsub2(unsigned int a, unsigned int b)
{
    unsigned int s, t;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vsub2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    s = a ^ b;          // sum bits
    t = a - b;          // actual sum
    s = s ^ t;          // determine carry-ins for each bit position
    s = s & 0x00010000; // borrow to high word 
    t = t + s;          // compensate for borrow from low word
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return t;           // halfword-wise difference
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubss2 (unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vsub2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    int ahi, alo, blo, bhi, rhi, rlo;
    ahi = (int)((a & 0xffff0000U));
    bhi = (int)((b & 0xffff0000U));
#if __CUDA_ARCH__ < 350
    asm ("prmt.b32 %0,%1,0,0x1044;" : "=r"(alo) : "r"(a));
    asm ("prmt.b32 %0,%1,0,0x1044;" : "=r"(blo) : "r"(b));
#else  /* __CUDA_ARCH__ < 350 */
    asm ("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(alo) : "r"(a));
    asm ("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(blo) : "r"(b));
#endif /* __CUDA_ARCH__ < 350 */
    asm ("sub.sat.s32 %0,%1,%2;" : "=r"(rlo) : "r"(alo), "r"(blo));
    asm ("sub.sat.s32 %0,%1,%2;" : "=r"(rhi) : "r"(ahi), "r"(bhi));
    asm ("prmt.b32 %0,%1,%2,0x7632;" : "=r"(r) : "r"(rlo), "r"(rhi));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise difference with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubus2 (unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vsub2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    int alo, blo, rlo, ahi, bhi, rhi;
    asm ("{                              \n\t"
         "and.b32     %0, %4, 0xffff;    \n\t"
         "and.b32     %1, %5, 0xffff;    \n\t"
#if __CUDA_ARCH__ < 350
         "shr.u32     %2, %4, 16;        \n\t"
         "shr.u32     %3, %5, 16;        \n\t"
#else  /* __CUDA_ARCH__ < 350 */
         "shf.r.clamp.b32  %2, %4, 0, 16;\n\t"
         "shf.r.clamp.b32  %3, %5, 0, 16;\n\t"
#endif /* __CUDA_ARCH__ < 350 */
         "}"
         : "=r"(alo), "=r"(blo), "=r"(ahi), "=r"(bhi) 
         : "r"(a), "r"(b));
    rlo = max ((int)(alo - blo), 0);
    rhi = max ((int)(ahi - bhi), 0);
    r = rhi * 65536 + rlo;
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise difference with unsigned saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vneg2(unsigned int a)
{
    return __vsub2 (0, a);// halfword-wise negation with wrap-around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vnegss2(unsigned int a)
{
    return __vsubss2(0,a);// halfword-wise negation with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    s = __vcmpges2 (a, b);// mask = 0xff if a >= b
    r = a ^ b;          //
    s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
    r = s ^ r;          // select a when b >= a, else select b => min(a,b)
    r = __vsub2 (s, r);   // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise absolute difference of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsads2(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm("vabsdiff2.s32.s32.s32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vabsdiffs2 (a, b);
    r = (s >> 16) + (s & 0x0000ffff);
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // halfword-wise sum of abs. differences of signed ints
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabs4(unsigned int a)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int b = 0, c = 0;
    asm ("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                       \n\t"
         ".reg .u32 a,m,r;        \n\t"
         "mov.b32  a,%1;          \n\t"
         "prmt.b32 m,a,0,0xba98;  \n\t" // msb ? 0xff : 00
         "xor.b32  r,a,m;         \n\t" // conditionally invert bits
         "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
         "add.u32  r,r,m;         \n\t" // conditionally add 1
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise absolute value, with wrap-around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsss4(unsigned int a)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int b = 0, c = 0;
    asm("vabsdiff4.s32.s32.s32.sat %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                       \n\t"
         ".reg .u32 a,m,r;        \n\t"
         "mov.b32  a,%1;          \n\t"
         "prmt.b32 m,a,0,0xba98;  \n\t" // msb ? 0xff : 00
         "xor.b32  r,a,m;         \n\t" // conditionally invert bits
         "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
         "add.u32  r,r,m;         \n\t" // conditionally add 1
         "prmt.b32 m,r,0,0xba98;  \n\t" // msb ? 0xff : 00
         "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
         "sub.u32  r,r,m;         \n\t" // subtract 1 if result wrapped around
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise absolute value with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vadd4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int r, c = 0;
    asm ("vadd4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    unsigned int r, s, t;
    s = a ^ b;          // sum bits
    r = a & 0x7f7f7f7f; // clear msbs
    t = b & 0x7f7f7f7f; // clear msbs
    s = s & 0x80808080; // msb sum bits
    r = r + t;          // add without msbs, record carry-out in msbs
    r = r ^ s;          // sum of msb sum and carry-in bits, w/o carry-out
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise sum, with wrap-around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddss4 (unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int r, c = 0;
    asm ("vadd4.sat.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /*
      For signed saturation, saturation is controlled by the overflow signal: 
      ovfl = (carry-in to msb) XOR (carry-out from msb). Overflow can only 
      occur when the msbs of both inputs are the same. The defined response to
      overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
      and 0x80 when the addends are negative (bit 7 set). The truth table for
      the msb is

      a   b   cy_in  res  cy_out  ovfl
      --------------------------------
      0   0       0    0       0     0
      0   0       1    1       0     1
      0   1       0    1       0     0
      0   1       1    0       1     0
      1   0       0    1       0     0
      1   0       1    0       1     0
      1   1       0    0       1     1
      1   1       1    1       1     0

      The seven low-order bits can be handled by simple wrapping addition with
      the carry out from bit 6 recorded in the msb (thus corresponding to the 
      cy_in in the truth table for the msb above). ovfl can be computed in many
      equivalent ways, here we use ovfl = (a ^ carry_in) & ~(a ^ b) since we 
      already need to compute (a ^ b) for the msb sum bit computation. First we
      compute the normal, wrapped addition result. When overflow is detected,
      we mask off the msb of the result, then compute a mask covering the seven
      low order bits, which are all set to 1. This sets the byte to 0x7f as we
      previously cleared the msb. In the overflow case, the sign of the result
      matches the sign of either of the inputs, so we extract the sign of a and
      add it to the low order bits, which turns 0x7f into 0x80, the correct 
      result for an overflowed negative result.
    */
    unsigned int r;
    asm ("{                         \n\t" 
         ".reg .u32 a,b,r,s,t,u;    \n\t"
         "mov.b32  a, %1;           \n\t" 
         "mov.b32  b, %2;           \n\t"
         "and.b32  r, a, 0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t, b, 0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32  s, a, b;         \n\t" // sum bits = (a ^ b)
         "add.u32  r, r, t;         \n\t" // capture msb carry-in in bit 7
         "xor.b32  t, a, r;         \n\t" // a ^ carry_in
         "not.b32  u, s;            \n\t" // ~(a ^ b)
         "and.b32  t, t, u;         \n\t" // ovfl = (a ^ carry_in) & ~(a ^ b)
         "and.b32  s, s, 0x80808080;\n\t" // msb sum bits
         "xor.b32  r, r, s;         \n\t" // msb result = (a ^ b ^ carry_in)
         "prmt.b32 s,a,0,0xba98;    \n\t" // sign(a) ? 0xff : 0
         "xor.b32  s,s,0x7f7f7f7f;  \n\t" // sign(a) ? 0x80 : 0x7f
         "prmt.b32 t,t,0,0xba98;    \n\t" // ovfl ? 0xff : 0
         "and.b32  s,s,t;           \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : 0
         "not.b32  t,t;             \n\t" // ~ovfl
         "and.b32  r,r,t;           \n\t" // ovfl ? 0 : a + b
         "or.b32   r,r,s;           \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : a+b
         "mov.b32  %0, r;           \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise sum with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddus4 (unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int r, c = 0;
    asm ("vadd4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // This code uses the same basic approach used for non-saturating addition.
    // The seven low-order bits in each byte are summed by regular addition,
    // with the carry-out from bit 6 (= carry-in for the msb) being recorded 
    // in bit 7, while the msb is handled separately.
    //
    // The fact that this is a saturating addition simplifies the handling of
    // the msb. When carry-out from the msb occurs, the entire byte must be
    // written as 0xff, and the computed msb is overwritten in the process. 
    // The corresponding entries in the truth table for the result msb thus 
    // become "don't cares":
    //
    // a  b  cy-in  res  cy-out
    // ------------------------
    // 0  0    0     0     0
    // 0  0    1     1     0
    // 0  1    0     1     0
    // 0  1    1     X     1
    // 1  0    0     1     0
    // 1  0    1     X     1
    // 1  1    0     X     1
    // 1  1    1     X     1
    //
    // As is easily seen, the simplest implementation of the result msb bit is 
    // simply (a | b | cy-in), with masking needed to isolate the msb. Note 
    // that this computation also makes the msb handling redundant with the 
    // clamping to 0xFF, because the msb is already set to 1 when saturation 
    // occurs. This means we only need to apply saturation to the seven lsb
    // bits in each byte, by overwriting with 0x7F. Saturation is controlled
    // by carry-out from the msb, which can be represented by various Boolean
    // expressions. Since to compute (a | b | cy-in) we need to compute (a | b)
    // anyhow, most efficient of these is cy-out = ((a & b) | cy-in) & (a | b).
    unsigned int r;
    asm ("{                         \n\t" 
         ".reg .u32 a,b,r,s,t,m;    \n\t"
         "mov.b32  a, %1;           \n\t" 
         "mov.b32  b, %2;           \n\t"
         "or.b32   m, a, b;         \n\t" // (a | b)
         "and.b32  r, a, 0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t, b, 0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  m, m, 0x80808080;\n\t" // (a | b), isolate msbs
         "add.u32  r, r, t;         \n\t" // add w/o msbs, record msb-carry-ins
         "and.b32  t, a, b;         \n\t" // (a & b)
         "or.b32   t, t, r;         \n\t" // (a & b) | cy-in)
         "or.b32   r, r, m;         \n\t" // msb = cy-in | (a | b)
         "and.b32  t, t, m;         \n\t" // cy-out=((a&b)|cy-in)&(a|b),in msbs
         "prmt.b32 t, t, 0, 0xba98; \n\t" // cy-out ? 0xff : 0
         "or.b32   r, r, t;         \n\t" // conditionally overwrite lsbs
         "mov.b32  %0, r;           \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // byte-wise sum with unsigned saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgs4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vavrg4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    // avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The 
    // two expressions can be re-written as follows to avoid needing additional
    // intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
    // ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
    // between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
    // ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
    // means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
    // depending on the sign bit of the shifted sum. By handling the msb sum 
    // bit of the result separately, we avoid carry-out during summation and
    // also can use (potentially faster) logical right shifts.
    asm ("{                      \n\t"
         ".reg .u32 a,b,c,r,s,t,u,v;\n\t"
         "mov.b32 a,%1;          \n\t" 
         "mov.b32 b,%2;          \n\t" 
         "and.b32 u,a,0xfefefefe;\n\t" // prevent shift crossing chunk boundary
         "and.b32 v,b,0xfefefefe;\n\t" // prevent shift crossing chunk boundary
         "xor.b32 s,a,b;         \n\t" // a ^ b
         "and.b32 t,a,b;         \n\t" // a & b
         "shr.u32 u,u,1;         \n\t" // a >> 1
         "shr.u32 v,v,1;         \n\t" // b >> 1
         "and.b32 c,s,0x01010101;\n\t" // (a ^ b) & 1
         "and.b32 s,s,0x80808080;\n\t" // extract msb (a ^ b)
         "and.b32 t,t,0x01010101;\n\t" // (a & b) & 1
         "add.u32 r,u,v;         \n\t" // (a>>1)+(b>>1) 
         "add.u32 r,r,t;         \n\t" // (a>>1)+(b>>1)+(a&b&1); rec. msb cy-in
         "xor.b32 r,r,s;         \n\t" // compute msb sum bit: a ^ b ^ cy-in
         "shr.u32 t,r,7;         \n\t" // sign ((a + b) >> 1)
         "not.b32 t,t;           \n\t" // ~sign ((a + b) >> 1)
         "and.b32 t,t,c;         \n\t" // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
         "add.u32 r,r,t;         \n\t" // conditionally add ((a ^ b) & 1)
         "mov.b32 %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise average of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vavrg4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
    // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
    c = a ^ b;           
    r = a | b;
    c = c & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
    c = c >> 1;
    r = r - c;
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise average of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vhaddu4(unsigned int a, unsigned int b)
{
    // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
    // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
    unsigned int r, s;   
    s = a ^ b;           
    r = a & b;
    s = s & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
    s = s >> 1;
    s = r + s;
    return s;           // byte-wise average of unsigned integers, rounded down
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpeq4(unsigned int a, unsigned int b)
{
    unsigned int c, r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    r = 0;
    asm ("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(c) : "r"(a), "r"(b), "r"(r));
    r = c << 8;         // convert bool
    r = r - c;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r & ~c;         // msb = 1, if r was 0x00
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// convert msbs to mask
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise (un)signed eq comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpges4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                          \n\t"
         ".reg .u32 a, b, r, s, t, u;\n\t"
         "mov.b32     a,%1;          \n\t" 
         "mov.b32     b,%2;          \n\t"
         "xor.b32     s,a,b;         \n\t" // a ^ b
         "or.b32      r,a,0x80808080;\n\t" // set msbs
         "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
         "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
         "xor.b32     r,r,s;         \n\t" // msb: ~sign(res) = a^b^~borrow-in
         "and.b32     t,t,s;         \n\t" // msb: ovfl= (~bw-in ^ a) & (a ^ b)
         "xor.b32     t,t,r;         \n\t" // msb: ge = ovfl != ~sign(res)
         "prmt.b32    r,t,0,0xba98;  \n\t" // build mask from msbs
         "mov.b32     %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed gt-eq comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vavgu4 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned gt-eq comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgts4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /* a <= b <===> a + ~b < 0 */
    asm ("{                       \n\t" 
         ".reg .u32 a,b,r,s,t,u;  \n\t"
         "mov.b32  a,%1;          \n\t" 
         "mov.b32  b,%2;          \n\t"
         "not.b32  b,b;           \n\t"
         "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32  s,a,b;         \n\t" // sum bits = (a ^ b)
         "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
         "xor.b32  t,a,r;         \n\t" // a ^ carry_in
         "not.b32  u,s;           \n\t" // ~(a ^ b)
         "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & ~(a^b)
         "xor.b32  r,r,u;         \n\t" // msb: ~result = (~(a ^ b) ^ carry_in)
         "xor.b32  t,t,r;         \n\t" // msb: gt = ovfl != sign(~res)
         "prmt.b32 r,t,0,0xba98;  \n\t" // build mask from msbs
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed gt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vhaddu4 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned gt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmples4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /* a <= b <===> a + ~b < 0 */
    asm ("{                       \n\t" 
         ".reg .u32 a,b,r,s,t,u;  \n\t"
         "mov.b32  a,%1;          \n\t" 
         "mov.b32  b,%2;          \n\t"
         "not.b32  u,b;           \n\t" // ~b
         "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t,u,0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32  u,a,b;         \n\t" // sum bits = (a ^ b)
         "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
         "xor.b32  t,a,r;         \n\t" // a ^ carry_in
         "not.b32  s,u;           \n\t" // ~(a ^ b)
         "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & (a ^ b)
         "xor.b32  r,r,s;         \n\t" // msb: result = (a ^ ~b ^ carry_in)
         "xor.b32  t,t,r;         \n\t" // msb: le = ovfl != sign(res)
         "prmt.b32 r,t,0,0xba98;  \n\t" // build mask from msbs
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed lt-eq comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpleu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vavgu4 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned lt-eq comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmplts4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                          \n\t"
         ".reg .u32 a, b, r, s, t, u;\n\t"
         "mov.b32     a,%1;          \n\t" 
         "mov.b32     b,%2;          \n\t"
         "not.b32     u,b;           \n\t" // ~b
         "xor.b32     s,u,a;         \n\t" // a ^ ~b
         "or.b32      r,a,0x80808080;\n\t" // set msbs
         "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
         "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
         "not.b32     u,s;           \n\t" // msb: ~(a^~b)
         "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
         "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
         "xor.b32     t,t,r;         \n\t" // msb: lt = ovfl != sign(res)
         "prmt.b32    r,t,0,0xba98;  \n\t" // build mask from msbs
         "mov.b32     %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed lt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpltu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vhaddu4 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // byte-wise unsigned lt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpne4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r | c;          // msb = 1, if r was not 0x00
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise (un)signed ne comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300
    s = 0;
    asm ("vabsdiff4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 */
    s = __vcmpgeu4 (a, b);// mask = 0xff if a >= b
    r = a ^ b;          //
    s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
    r = s ^ r;          // select a when b >= a, else select b => min(a,b)
    r = s - r;          // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 */
    return r;           // byte-wise absolute difference of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxs4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vcmpges4 (a, b);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise maximum of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmax4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vcmpgeu4 (a, b);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise maximum of unsigned integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmins4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmin4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vcmpges4 (b, a);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise minimum of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vminu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vmin4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vcmpgeu4 (b, a);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise minimum of unsigned integers
}
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vseteq4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r & ~c;         // msb = 1, if r was 0x00
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise (un)signed eq comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetles4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /* a <= b <===> a + ~b < 0 */
    asm ("{                       \n\t" 
         ".reg .u32 a,b,r,s,t,u;  \n\t"
         "mov.b32  a,%1;          \n\t" 
         "mov.b32  b,%2;          \n\t"
         "not.b32  u,b;           \n\t" // ~b
         "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t,u,0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32  u,a,b;         \n\t" // sum bits = (a ^ b)
         "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
         "xor.b32  t,a,r;         \n\t" // a ^ carry_in
         "not.b32  s,u;           \n\t" // ~(a ^ b)
         "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & (a ^ b)
         "xor.b32  r,r,s;         \n\t" // msb: result = (a ^ ~b ^ carry_in)
         "xor.b32  t,t,r;         \n\t" // msb: le = ovfl != sign(res)
         "and.b32  t,t,0x80808080;\n\t" // isolate msbs
         "shr.u32  r,t,7;         \n\t" // convert to bool
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed lt-eq comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetleu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vavgu4 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned lt-eq comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetlts4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    asm ("{                          \n\t"
         ".reg .u32 a, b, r, s, t, u;\n\t"
         "mov.b32     a,%1;          \n\t" 
         "mov.b32     b,%2;          \n\t"
         "not.b32     u,b;           \n\t" // ~b
         "or.b32      r,a,0x80808080;\n\t" // set msbs
         "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32     s,u,a;         \n\t" // a ^ ~b
         "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
         "not.b32     u,s;           \n\t" // msb: ~(a^~b)
         "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
         "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
         "xor.b32     t,t,r;         \n\t" // msb: lt = ovfl != sign(res)
         "and.b32     t,t,0x80808080;\n\t" // isolate msbs
         "shr.u32     r,t,7;         \n\t" // convert to bool
         "mov.b32     %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed lt comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetltu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(a));
    c = __vhaddu4 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned lt comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetges4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("{                          \n\t"
         ".reg .u32 a, b, r, s, t, u;\n\t"
         "mov.b32     a,%1;          \n\t" 
         "mov.b32     b,%2;          \n\t"
         "xor.b32     s,a,b;         \n\t" // a ^ b
         "or.b32      r,a,0x80808080;\n\t" // set msbs
         "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
         "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
         "xor.b32     r,r,s;         \n\t" // msb: ~sign(res) = a^b^~borrow-in
         "and.b32     t,t,s;         \n\t" // msb: ovfl= (~bw-in ^ a) & (a ^ b)
         "xor.b32     t,t,r;         \n\t" // msb: ge = ovfl != ~sign(res)
         "and.b32     t,t,0x80808080;\n\t" // isolate msbs
         "shr.u32     r,t,7;         \n\t" // convert to bool
         "mov.b32     %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise signed gt-eq comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgeu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vavgu4 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned gt-eq comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgts4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /* a <= b <===> a + ~b < 0 */
    asm ("{                       \n\t" 
         ".reg .u32 a,b,r,s,t,u;  \n\t"
         "mov.b32  a,%1;          \n\t" 
         "mov.b32  b,%2;          \n\t"
         "not.b32  b,b;           \n\t"
         "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
         "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
         "xor.b32  s,a,b;         \n\t" // sum bits = (a ^ b)
         "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
         "xor.b32  t,a,r;         \n\t" // a ^ carry_in
         "not.b32  u,s;           \n\t" // ~(a ^ b)
         "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & ~(a^b)
         "xor.b32  r,r,u;         \n\t" // msb: ~result = (~(a ^ b) ^ carry_in)
         "xor.b32  t,t,r;         \n\t" // msb: gt = ovfl != sign(~res)
         "and.b32  t,t,0x80808080;\n\t" // isolate msbs
         "shr.u32  r,t,7;         \n\t" // convert to bool
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */ 
    return r;           // byte-wise signed gt comparison with mask result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgtu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    asm ("not.b32 %0,%0;" : "+r"(b));
    c = __vhaddu4 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise unsigned gt comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetne4(unsigned int a, unsigned int b)
{
    unsigned int r, c;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    c = 0;
    asm ("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r | c;          // msb = 1, if r was not 0x00
    c = c & 0x80808080; // extract msbs
    r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise (un)signed ne comparison with bool result
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsadu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300
    s = 0;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 */
    r = __vabsdiffu4 (a, b);
    s = r >> 8;
    r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
    r = ((r << 16) + r) >> 16;
#endif /*  __CUDA_ARCH__ >= 300 */
    return r;           // byte-wise sum of absol. differences of unsigned ints
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsub4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int r, c = 0;
    asm ("vsub4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500*/
    unsigned int r, s, t;
    s = a ^ ~b;         // inverted sum bits
    r = a | 0x80808080; // set msbs
    t = b & 0x7f7f7f7f; // clear msbs
    s = s & 0x80808080; // inverted msb sum bits
    r = r - t;          // subtract w/o msbs, record inverted borrows in msb
    r = r ^ s;          // combine inverted msb sum bits and borrows
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise difference
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubss4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    /*
      For signed saturation, saturation is controlled by the overflow signal: 
      ovfl = (borrow-in to msb) XOR (borrow-out from msb). Overflow can only 
      occur when the msbs of both inputs are differemt. The defined response to
      overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
      and 0x80 when the addends are negative (bit 7 set). The truth table for
      the msb is

      a   b  bw_in  res  bw_out  ovfl  a^~bw_in  ~(a^~b) (a^~bw_in)&~(a^~b)
      ---------------------------------------------------------------------
      0   0      0    0       0     0         1        0                  0
      0   0      1    1       1     0         0        0                  0
      0   1      0    1       1     1         1        1                  1
      0   1      1    0       1     0         0        1                  0
      1   0      0    1       0     0         0        1                  0
      1   0      1    0       0     1         1        1                  1
      1   1      0    0       0     0         0        0                  0
      1   1      1    1       1     0         1        0                  0

      The seven low-order bits can be handled by wrapping subtraction with the
      borrow-out from bit 6 recorded in the msb (thus corresponding to the 
      bw_in in the truth table for the msb above). ovfl can be computed in many
      equivalent ways, here we use ovfl = (a ^ ~borrow_in) & ~(a ^~b) since we 
      already need to compute (a ^~b) and ~borrow-in for the msb result bit 
      computation. First we compute the normal, wrapped subtraction result. 
      When overflow is detected, we mask off the result's msb, then compute a
      mask covering the seven low order bits, which are all set to 1. This sets
      the byte to 0x7f as we previously cleared the msb. In the overflow case, 
      the sign of the result matches the sign of input a, so we extract the 
      sign of a and add it to the low order bits, which turns 0x7f into 0x80, 
      the correct result for an overflowed negative result.
    */
    asm ("{                          \n\t"
         ".reg .u32 a,b,r,s,t,u,v,w; \n\t"
         "mov.b32     a,%1;          \n\t" 
         "mov.b32     b,%2;          \n\t"
         "not.b32     u,b;           \n\t" // ~b
         "xor.b32     s,u,a;         \n\t" // a ^ ~b
         "or.b32      r,a,0x80808080;\n\t" // set msbs
         "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
         "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
         "not.b32     u,s;           \n\t" // msb: ~(a^~b)
         "and.b32     s,s,0x80808080;\n\t" // msb: a ^ ~b
         "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
         "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
         "prmt.b32    s,a,0,0xba98;  \n\t" // sign(a) ? 0xff : 0
         "xor.b32     s,s,0x7f7f7f7f;\n\t" // sign(a) ? 0x80 : 0x7f
         "prmt.b32    t,t,0,0xba98;  \n\t" // ovfl ? 0xff : 0
         "and.b32     s,s,t;         \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : 0
         "not.b32     t,t;           \n\t" // ~ovfl
         "and.b32     r,r,t;         \n\t" // ovfl ? 0 : a + b
         "or.b32      r,r,s;         \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) :a+b
         "mov.b32     %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise difference with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubus4(unsigned int a, unsigned int b)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int c = 0;
    asm ("vsub4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    // This code uses the same basic approach used for the non-saturating 
    // subtraction. The seven low-order bits in each byte are subtracted by 
    // regular subtraction with the inverse of the borrow-out from bit 6 (= 
    // inverse of borrow-in for the msb) being recorded in bit 7, while the 
    // msb is handled separately.
    //
    // Clamping to 0 needs happens when there is a borrow-out from the msb.
    // This is simply accomplished by ANDing the normal addition result with
    // a mask based on the inverted msb borrow-out: ~borrow-out ? 0xff : 0x00.
    // The borrow-out information is generated from the msb. Since we already 
    // have the msb's ~borrow-in and (a^~b) available from the computation of
    // the msb result bit, the most efficient way to compute msb ~borrow-out 
    // is: ((a ^ ~b) & ~borrow-in) | (~b & a). The truth table for the msb is
    //
    // a b bw-in res ~bw-out a^~b (a^~b)&~bw-in (a&~b) ((a^~b)&~bw-in)|(a&~b)
    //                                                        
    // 0 0  0     0     1      1        1          0          1
    // 0 0  1     1     0      1        0          0          0
    // 0 1  0     1     0      0        0          0          0
    // 0 1  1     0     0      0        0          0          0
    // 1 0  0     1     1      0        0          1          1
    // 1 0  1     0     1      0        0          1          1
    // 1 1  0     0     1      1        1          0          1
    // 1 1  1     1     0      1        0          0          0
    //
    asm ("{                       \n\t"
         ".reg .u32 a,b,r,s,t,u;  \n\t"
         "mov.b32  a,%1;          \n\t"
         "mov.b32  b,%2;          \n\t"
         "not.b32  u,b;           \n\t" // ~b
         "xor.b32  s,u,a;         \n\t" // a ^ ~b
         "and.b32  u,u,a;         \n\t" // a & ~b
         "or.b32   r,a,0x80808080;\n\t" // set msbs
         "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
         "sub.u32  r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
         "and.b32  t,r,s;         \n\t" // msb: (a ^ ~b) & ~borrow-in
         "and.b32  s,s,0x80808080;\n\t" // msb: a ^ ~b
         "xor.b32  r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
         "or.b32   t,t,u;         \n\t" // msb: bw-out = ((a^~b)&~bw-in)|(a&~b)
         "prmt.b32 t,t,0,0xba98;  \n\t" // ~borrow-out ? 0xff : 0
         "and.b32  r,r,t;         \n\t" // cond. clear result if msb borrow-out
         "mov.b32  %0,r;          \n\t"
         "}"
         : "=r"(r) : "r"(a) , "r"(b));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise difference with unsigned saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vneg4(unsigned int a)
{
    return __vsub4 (0, a);// byte-wise negation with wrap-around
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vnegss4(unsigned int a)
{
    unsigned int r;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    unsigned int s = 0;
    asm ("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(s),"r"(a),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    r = __vsub4 (0, a);   //
    asm ("{                       \n\t"
         ".reg .u32 a, r, s;      \n\t"
         "mov.b32  r,%0;          \n\t"
         "mov.b32  a,%1;          \n\t"
         "and.b32  a,a,0x80808080;\n\t" // extract msb
         "and.b32  s,a,r;         \n\t" // wrap-around if msb set in a and -a
         "shr.u32  s,s,7;         \n\t" // msb ? 1 : 0
         "sub.u32  r,r,s;         \n\t" // subtract 1 if result wrapped around
         "mov.b32  %0,r;          \n\t"
         "}"
         : "+r"(r) : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise negation with signed saturation
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm ("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    s = __vcmpges4 (a, b);// mask = 0xff if a >= b
    r = a ^ b;          //
    s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
    r = s ^ r;          // select a when b >= a, else select b => min(a,b)
    r = __vsub4 (s, r);   // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise absolute difference of signed integers
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsads4(unsigned int a, unsigned int b)
{
    unsigned int r, s;
#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
    s = 0;
    asm("vabsdiff4.s32.s32.s32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));
#else  /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    r = __vabsdiffs4 (a, b);
    s = r >> 8;
    r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
    r = ((r << 16) + r) >> 16;
#endif /* __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500 */
    return r;           // byte-wise sum of absolute differences of signed ints
}

/*******************************************************************************
 *                                                                             *
 *                             END SIMD functions                              *
 *                                                                             *
 *******************************************************************************/
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef __DEVICE_FUNCTIONS_DECL__
#undef __DEVICE_FUNCTIONS_STATIC_DECL__

#endif /* __CUDACC_RTC__ || __CUDABE__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* !__DEVICE_FUNCTIONS_HPP__ */

