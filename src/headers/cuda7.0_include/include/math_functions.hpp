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

#if !defined(__MATH_FUNCTIONS_HPP__)
#define __MATH_FUNCTIONS_HPP__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__) || defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__)

__host__ __device__ __cudart_builtin__ int signbit(float x) { return __signbitf(x); }
__host__ __device__ __cudart_builtin__ int signbit(double x) { return __signbit(x); }
__host__ __device__ __cudart_builtin__ int signbit(long double x) { return __signbitl((double)x);}

__host__ __device__ __cudart_builtin__ int isfinite(float x) { return __finitef(x); } 
__host__ __device__ __cudart_builtin__ int isfinite(double x) { return __finite(x); }
__host__ __device__ __cudart_builtin__ int isfinite(long double x) { return __finitel((double)x); }

__host__ __device__ __cudart_builtin__ int isnan(float x) { return __isnanf(x); }
__host__ __device__ __cudart_builtin__ int isnan(double x) { return __isnan(x); }
__host__ __device__ __cudart_builtin__ int isnan(long double x) { return __isnanl((double)x); }

__host__ __device__ __cudart_builtin__ int isinf(float x) { return __isinff(x); }
__host__ __device__ __cudart_builtin__ int isinf(double x) { return __isinf(x); }
__host__ __device__ __cudart_builtin__ int isinf(long double x) { return __isinfl((double)x); }

__host__ __device__ __cudart_builtin__ long long int abs(long long int a) { return llabs(a); }

__host__ __device__ __cudart_builtin__ long int  abs(long int in)        { return llabs(in); }
__host__ __device__ __cudart_builtin__ float     abs(float in)           { return fabsf(in); }
__host__ __device__ __cudart_builtin__ double    abs(double in)          { return fabs(in); }
__host__ __device__ __cudart_builtin__ float     fabs(float in)          { return fabsf(in); }
__host__ __device__ __cudart_builtin__ float     ceil(float in)          { return ceilf(in); }
__host__ __device__ __cudart_builtin__ float     floor(float in)         { return floorf(in); }
__host__ __device__ __cudart_builtin__ float     sqrt(float in)          { return sqrtf(in); }
__host__ __device__ __cudart_builtin__ float     pow(float a, float b)   { return powf(a, b); }
__host__ __device__ float powif(float, int); 
__host__ __device__ __cudart_builtin__ float     pow(float a, int b)     { return powif(a, b); }
__host__ __device__ double powi(double, int);
__host__ __device__ __cudart_builtin__ double    pow(double a, int b)    { return powi(a, b); }
__host__ __device__ __cudart_builtin__ float     log(float in)           { return logf(in); }
__host__ __device__ __cudart_builtin__ float     log10(float in)         { return log10f(in); }
__host__ __device__ __cudart_builtin__ float     fmod(float a, float b)  { return fmodf(a, b); }
__host__ __device__ __cudart_builtin__ float     modf(float a, float*b)  { return modff(a, b); }
__host__ __device__ __cudart_builtin__ float     exp(float in)           { return expf(in); }
__host__ __device__ __cudart_builtin__ float     frexp(float a, int*b)   { return frexpf(a, b); }
__host__ __device__ __cudart_builtin__ float     ldexp(float a, int b)   { return ldexpf(a, b); }
__host__ __device__ __cudart_builtin__ float     asin(float in)          { return asinf(in); }
__host__ __device__ __cudart_builtin__ float     sin(float in)           { return sinf(in); }
__host__ __device__ __cudart_builtin__ float     sinh(float in)          { return sinhf(in); }
__host__ __device__ __cudart_builtin__ float     acos(float in)          { return acosf(in); }
__host__ __device__ __cudart_builtin__ float     cos(float in)           { return cosf(in); }
__host__ __device__ __cudart_builtin__ float     cosh(float in)          { return coshf(in); }
__host__ __device__ __cudart_builtin__ float     atan(float in)          { return atanf(in); }
__host__ __device__ __cudart_builtin__ float     atan2(float a, float b) { return atan2f(a, b); }
__host__ __device__ __cudart_builtin__ float     tan(float in)           { return tanf(in); }
__host__ __device__ __cudart_builtin__ float     tanh(float in)          { return tanhf(in); }

#elif defined(__GNUC__)

#undef signbit
#undef isfinite
#undef isnan
#undef isinf

#if defined(__APPLE__)
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(float x) { return __signbitf(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x) { return __signbitd(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(long double x) { return __signbitl(x);}

__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(float x) { return __isfinitef(x); } 
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x) { return __isfinited(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(long double x) { return __isfinite(x); }

__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(float x) { return __isnanf(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x) throw()  { return __isnand(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(long double x) { return __isnan(x); }

__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(float x) { return __isinff(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x) throw()  { return __isinfd(x); }
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(long double x) { return __isinf(x); }

#if defined(_LIBCPP_VERSION)
extern "C" __host__ __device__ float powif(float, int);
__forceinline__ __host__ __device__ __cudart_builtin__ float pow(float a, int b) throw()
{
#if defined(__CUDA_ARCH__)
  return powif(a, (b));
#else /* !defined(__CUDA_ARCH__) */
  return pow<float, int>(a, b);
#endif /* defined(__CUDA_ARCH__) */
}
extern "C" __host__ __device__ double powi(double, int);
__forceinline__ __host__ __device__ __cudart_builtin__ double pow(double a, int b) throw()
{
#if defined(__CUDA_ARCH__)
  return powi(a, (b)); 
#else /* !defined(__CUDA_ARCH__) */
  return pow<double, int>(a, b);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _LIBCPP_VERSION */

#else /* __APPLE__ */


#if defined(__QNX__)
extern "C" __host__ __device__ float powif(float, int); 
__forceinline__ __host__ __device__ __cudart_builtin__ float     pow(float a, int b)     { return powif(a, b); }
extern "C" __host__ __device__ double powi(double, int); 
__forceinline__ __host__ __device__ __cudart_builtin__ double pow(double a, int b)     { return powi(a, b); }

static __inline__ __host__ __device__ bool isfinite(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__finitel(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isfinite(double a)
{
#if defined(__CUDA_ARCH__)
  return (__finite(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isfinite(float a)
{
#if defined(__CUDA_ARCH__)
  return (__finitef(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}

static __inline__ __host__ __device__ bool isnan(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__isnanl(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isnan(double a)
{
#if defined(__CUDA_ARCH__)
  return (__isnan(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isnan(float a)
{
#if defined(__CUDA_ARCH__)
  return (__isnanf(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}

static __inline__ __host__ __device__ bool isinf(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__isinfl(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isinf(double a)
{
#if defined(__CUDA_ARCH__)
  return (__isinf(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
static __inline__ __host__ __device__ bool isinf(float a)
{
#if defined(__CUDA_ARCH__)
  return (__isinff(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}

#else /* !QNX */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(float x) { return __signbitf(x); }
#if defined(__ICC)
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x) throw() { return __signbit(x); }
#else /* !__ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x) { return __signbit(x); }
#endif /* __ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(long double x) { return __signbitl(x);}

__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(float x) { return __finitef(x); } 
#if defined(__ICC)
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x) throw() { return __finite(x); }
#else /* !__ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x) { return __finite(x); }
#endif /* __ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(long double x) { return __finitel(x); }


__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(float x) { return __isnanf(x); }
#if defined(__ANDROID__)
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x) { return __isnan(x); }
#else /* !__ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x) throw()  { return __isnan(x); }
#endif /* __ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(long double x) { return __isnanl(x); }

__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(float x) { return __isinff(x); }
#if defined(__ANDROID__)
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x) { return __isinf(x); }
#else /* !__ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x) throw()  { return __isinf(x); }
#endif /* __ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(long double x) { return __isinfl(x); }
#endif /* QNX */

#endif /* __APPLE__ */

#if defined(__arm__) && !defined(_STLPORT_VERSION) && !_GLIBCXX_USE_C99
#if !defined(__ANDROID__) || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)

#if !defined(__QNX__)
static __inline__ __host__ __device__ __cudart_builtin__ long long int abs(long long int a)
{
  return llabs(a);
}
#endif /* !QNX */

#endif /* !__ANDROID__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8) */
#endif /* __arm__ && !_STLPORT_VERSION && !_GLIBCXX_USE_C99 */

#elif defined(_WIN32)

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int signbit(long double a)
{
  return __signbitl(a);
}

static __inline__ __host__ __device__ int signbit(double a)
{
  return __signbit(a);
}

static __inline__ __host__ __device__ int signbit(float a)
{
  return __signbitf(a);
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isinf(long double a)
{
  return __isinfl(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isinf(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__isinfl(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isinf(double a)
{
  return __isinf(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isinf(double a)
{
#if defined(__CUDA_ARCH__)
  return (__isinf(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isinf(float a)
{
  return __isinff(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isinf(float a)
{
#if defined(__CUDA_ARCH__)
  return (__isinff(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isinf<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isnan(long double a)
{
  return __isnanl(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isnan(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__isnanl(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isnan(double a)
{
  return __isnan(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isnan(double a)
{
#if defined(__CUDA_ARCH__)
  return (__isnan(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isnan(float a)
{
  return __isnanf(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isnan(float a)
{
#if defined(__CUDA_ARCH__)
  return (__isnanf(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isnan<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isfinite(long double a)
{
  return __finitel(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isfinite(long double a)
{
#if defined(__CUDA_ARCH__)
  return (__finitel(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<long double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isfinite(double a)
{
  return __finite(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isfinite(double a)
{
#if defined(__CUDA_ARCH__)
  return (__finite(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<double>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isfinite(float a)
{
  return __finitef(a);
}
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isfinite(float a)
{
#if defined(__CUDA_ARCH__)
  return (__finitef(a) != 0);
#else /* defined(__CUDA_ARCH__) */
  return isfinite<float>(a);
#endif /* defined(__CUDA_ARCH__) */
}
#endif /* _MSC_VER < 1800 */

#endif /* __CUDACC_RTC__ */

#if defined(__CUDACC_RTC__)
#define __MATH_FUNCTIONS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline __host__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__CUDACC_RTC__) || _MSC_VER < 1800
#if !defined(__QNX__)
__MATH_FUNCTIONS_DECL__ float logb(float a)
{
  return logbf(a);
}

__MATH_FUNCTIONS_DECL__ int ilogb(float a)
{
  return ilogbf(a);
}

__MATH_FUNCTIONS_DECL__ float scalbn(float a, int b)
{
  return scalbnf(a, b);
}

__MATH_FUNCTIONS_DECL__ float scalbln(float a, long int b)
{
  return scalblnf(a, b);
}

__MATH_FUNCTIONS_DECL__ float exp2(float a)
{
  return exp2f(a);
}

__MATH_FUNCTIONS_DECL__ float expm1(float a)
{
  return expm1f(a);
}

__MATH_FUNCTIONS_DECL__ float log2(float a)
{
  return log2f(a);
}

__MATH_FUNCTIONS_DECL__ float log1p(float a)
{
  return log1pf(a);
}

__MATH_FUNCTIONS_DECL__ float acosh(float a)
{
  return acoshf(a);
}

__MATH_FUNCTIONS_DECL__ float asinh(float a)
{
  return asinhf(a);
}

__MATH_FUNCTIONS_DECL__ float atanh(float a)
{
  return atanhf(a);
}

__MATH_FUNCTIONS_DECL__ float hypot(float a, float b)
{
  return hypotf(a, b);
}

__MATH_FUNCTIONS_DECL__ float norm3d(float a, float b, float c)
{
  return norm3df(a, b, c);
}

__MATH_FUNCTIONS_DECL__ float norm4d(float a, float b, float c, float d)
{
  return norm4df(a, b, c, d);
}

__MATH_FUNCTIONS_DECL__ float cbrt(float a)
{
  return cbrtf(a);
}

__MATH_FUNCTIONS_DECL__ float erf(float a)
{
  return erff(a);
}

__MATH_FUNCTIONS_DECL__ float erfc(float a)
{
  return erfcf(a);
}

__MATH_FUNCTIONS_DECL__ float lgamma(float a)
{
  return lgammaf(a);
}

__MATH_FUNCTIONS_DECL__ float tgamma(float a)
{
  return tgammaf(a);
}

__MATH_FUNCTIONS_DECL__ float copysign(float a, float b)
{
  return copysignf(a, b);
}

__MATH_FUNCTIONS_DECL__ float nextafter(float a, float b)
{
  return nextafterf(a, b);
}

__MATH_FUNCTIONS_DECL__ float remainder(float a, float b)
{
  return remainderf(a, b);
}

__MATH_FUNCTIONS_DECL__ float remquo(float a, float b, int *quo)
{
  return remquof(a, b, quo);
}

__MATH_FUNCTIONS_DECL__ float round(float a)
{
  return roundf(a);
}

__MATH_FUNCTIONS_DECL__ long int lround(float a)
{
  return lroundf(a);
}

__MATH_FUNCTIONS_DECL__ long long int llround(float a)
{
  return llroundf(a);
}

__MATH_FUNCTIONS_DECL__ float trunc(float a)
{
  return truncf(a);
}

__MATH_FUNCTIONS_DECL__ float rint(float a)
{
  return rintf(a);
}

__MATH_FUNCTIONS_DECL__ long int lrint(float a)
{
  return lrintf(a);
}

__MATH_FUNCTIONS_DECL__ long long int llrint(float a)
{
  return llrintf(a);
}

__MATH_FUNCTIONS_DECL__ float nearbyint(float a)
{
  return nearbyintf(a);
}

__MATH_FUNCTIONS_DECL__ float fdim(float a, float b)
{
  return fdimf(a, b);
}

__MATH_FUNCTIONS_DECL__ float fma(float a, float b, float c)
{
  return fmaf(a, b, c);
}

__MATH_FUNCTIONS_DECL__ float fmax(float a, float b)
{
  return fmaxf(a, b);
}

__MATH_FUNCTIONS_DECL__ float fmin(float a, float b)
{
  return fminf(a, b);
}
#endif /* !QNX */
#endif /* __CUDACC__JIT__ || _MSC_VER < 1800 */

__MATH_FUNCTIONS_DECL__ float exp10(float a)
{
  return exp10f(a);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float a)
{
  return rsqrtf(a);
}

__MATH_FUNCTIONS_DECL__ float rcbrt(float a)
{
  return rcbrtf(a);
}

__MATH_FUNCTIONS_DECL__ float sinpi(float a)
{
  return sinpif(a);
}

__MATH_FUNCTIONS_DECL__ float cospi(float a)
{
  return cospif(a);
}

__MATH_FUNCTIONS_DECL__ void sincospi(float a, float *sptr, float *cptr)
{
  sincospif(a, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ void sincos(float a, float *sptr, float *cptr)
{
  sincosf(a, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ float j0(float a)
{
  return j0f(a);
}

__MATH_FUNCTIONS_DECL__ float j1(float a)
{
  return j1f(a);
}

__MATH_FUNCTIONS_DECL__ float jn(int n, float a)
{
  return jnf(n, a);
}

__MATH_FUNCTIONS_DECL__ float y0(float a)
{
  return y0f(a);
}

__MATH_FUNCTIONS_DECL__ float y1(float a)
{
  return y1f(a);
}

__MATH_FUNCTIONS_DECL__ float yn(int n, float a)
{ 
  return ynf(n, a);
}

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i0(float a)
{
  return cyl_bessel_i0f(a);
}

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i1(float a)
{
  return cyl_bessel_i1f(a);
}

__MATH_FUNCTIONS_DECL__ float erfinv(float a)
{
  return erfinvf(a);
}

__MATH_FUNCTIONS_DECL__ float erfcinv(float a)
{
  return erfcinvf(a);
}

__MATH_FUNCTIONS_DECL__ float normcdfinv(float a)
{
  return normcdfinvf(a);
}

__MATH_FUNCTIONS_DECL__ float normcdf(float a)
{
  return normcdff(a);
}

__MATH_FUNCTIONS_DECL__ float erfcx(float a)
{
  return erfcxf(a);
}

__MATH_FUNCTIONS_DECL__ double copysign(double a, float b)
{
  return copysign(a, (double)b);
}

__MATH_FUNCTIONS_DECL__ float copysign(float a, double b)
{
  return copysignf(a, (float)b);
}

__MATH_FUNCTIONS_DECL__ unsigned int min(unsigned int a, unsigned int b)
{
  return umin(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned int min(int a, unsigned int b)
{
  return umin((unsigned int)a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned int min(unsigned int a, int b)
{
  return umin(a, (unsigned int)b);
}

__MATH_FUNCTIONS_DECL__ long long int min(long long int a, long long int b)
{
  return llmin(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int min(unsigned long long int a, unsigned long long int b)
{
  return ullmin(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int min(long long int a, unsigned long long int b)
{
  return ullmin((unsigned long long int)a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int min(unsigned long long int a, long long int b)
{
  return ullmin(a, (unsigned long long int)b);
}

__MATH_FUNCTIONS_DECL__ float min(float a, float b)
{
  return fminf(a, b);
}

__MATH_FUNCTIONS_DECL__ double min(double a, double b)
{
  return fmin(a, b);
}

__MATH_FUNCTIONS_DECL__ double min(float a, double b)
{
  return fmin((double)a, b);
}

__MATH_FUNCTIONS_DECL__ double min(double a, float b)
{
  return fmin(a, (double)b);
}

__MATH_FUNCTIONS_DECL__ unsigned int max(unsigned int a, unsigned int b)
{
  return umax(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned int max(int a, unsigned int b)
{
  return umax((unsigned int)a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned int max(unsigned int a, int b)
{
  return umax(a, (unsigned int)b);
}

__MATH_FUNCTIONS_DECL__ long long int max(long long int a, long long int b)
{
  return llmax(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int max(unsigned long long int a, unsigned long long int b)
{
  return ullmax(a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int max(long long int a, unsigned long long int b)
{
  return ullmax((unsigned long long int)a, b);
}

__MATH_FUNCTIONS_DECL__ unsigned long long int max(unsigned long long int a, long long int b)
{
  return ullmax(a, (unsigned long long int)b);
}

__MATH_FUNCTIONS_DECL__ float max(float a, float b)
{
  return fmaxf(a, b);
}

__MATH_FUNCTIONS_DECL__ double max(double a, double b)
{
  return fmax(a, b);
}

__MATH_FUNCTIONS_DECL__ double max(float a, double b)
{
  return fmax((double)a, b);
}

__MATH_FUNCTIONS_DECL__ double max(double a, float b)
{
  return fmax(a, (double)b);
}

#undef __MATH_FUNCTIONS_DECL__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* __CUDACC_RTC__ || __cplusplus && __CUDACC__ */
#if defined(__CUDACC_RTC__) || !defined(__CUDACC__)

#include "host_defines.h"
#include "math_constants.h"

#define __cuda_INT_MAX \
        ((int)((unsigned int)-1 >> 1))

#if defined(__CUDACC_RTC__) || defined(__CUDABE__)

#if !defined(__CUDACC_RTC__)
#define EXCLUDE_FROM_RTC
#include "device_functions_decls.h"
#undef EXCLUDE_FROM_RTC
#endif /* !__CUDACC_RTC__ */
#include "device_functions.h"

#if defined(__CUDACC_RTC__)
#define __MATH_FUNCTIONS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static __forceinline__
#endif /* __CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
* DEVICE                                                                       *
*                                                                              *
*******************************************************************************/

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DECL__ float rintf(float a)
{
  return __nv_rintf(a);
}

__MATH_FUNCTIONS_DECL__ long int lrintf(float a)
{
#if defined(__LP64__)
  return (long int)__float2ll_rn(a);
#else /* __LP64__ */
  return (long int)__float2int_rn(a);
#endif /* __LP64__ */
}

__MATH_FUNCTIONS_DECL__ long long int llrintf(float a)
{
  return __nv_llrintf(a);
}

__MATH_FUNCTIONS_DECL__ float nearbyintf(float a)
{
  return __nv_nearbyintf(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DECL__ int __signbitf(float a)
{
  return __nv_signbitf(a);
}

#if _MSC_VER >= 1800
__MATH_FUNCTIONS_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a);
__MATH_FUNCTIONS_DECL__ int _ldsign(/* we do not support long double yet, hence double */double a)
{
  return __signbitl(a);
}

__MATH_FUNCTIONS_DECL__ int __signbit(double a);
__MATH_FUNCTIONS_DECL__ int _dsign(double a)
{
  return __signbit(a);
}

__MATH_FUNCTIONS_DECL__ int _fdsign(float a)
{
  return __signbitf(a);
}
#endif

__MATH_FUNCTIONS_DECL__ float copysignf(float a, float b)
{
  return __nv_copysignf(a, b);
}

__MATH_FUNCTIONS_DECL__ int __finitef(float a)
{
  return __nv_finitef(a);
}

#if defined(__APPLE__)

__MATH_FUNCTIONS_DECL__ int __isfinitef(float a)
{
  return __finitef(a);
}

#endif /* __APPLE__ */

__MATH_FUNCTIONS_DECL__ int __isinff(float a)
{
  return __nv_isinff(a);
}

__MATH_FUNCTIONS_DECL__ int __isnanf(float a)
{
  return __nv_isnanf(a);
}

__MATH_FUNCTIONS_DECL__ float nextafterf(float a, float b)
{
  return __nv_nextafterf(a, b);
}

__MATH_FUNCTIONS_DECL__ float nanf(const char *tagp)
{
  return __nv_nanf((const signed char *) tagp);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DECL__ float sinf(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_sinf(a);
#else /* __USE_FAST_MATH__ */
  return __nv_sinf(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float cosf(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_cosf(a);
#else /* __USE_FAST_MATH__ */
  return __nv_cosf(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ void sincosf(float a, float *sptr, float *cptr)
{
#if defined(__USE_FAST_MATH__)
  __nv_fast_sincosf(a, sptr, cptr);
#else /* __USE_FAST_MATH__ */
  __nv_sincosf(a, sptr, cptr);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float sinpif(float a)
{
  return __nv_sinpif(a);
}

__MATH_FUNCTIONS_DECL__ float cospif(float a)
{
  return __nv_cospif(a);
}

__MATH_FUNCTIONS_DECL__ void sincospif(float a, float *sptr, float *cptr)
{
  __nv_sincospif(a, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ float tanf(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_tanf(a);
#else /* __USE_FAST_MATH__ */
  return __nv_tanf(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float log2f(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_log2f(a);
#else /* __USE_FAST_MATH__ */
  return __nv_log2f(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float expf(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_expf(a);
#else /* __USE_FAST_MATH__ */
  return __nv_expf(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float exp10f(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_exp10f(a);
#else /* __USE_FAST_MATH__ */
  return __nv_exp10f(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float coshf(float a)
{
  return __nv_coshf(a);
}

__MATH_FUNCTIONS_DECL__ float sinhf(float a)
{
  return __nv_sinhf(a);
}

__MATH_FUNCTIONS_DECL__ float tanhf(float a)
{
  return __nv_tanhf(a);
}

__MATH_FUNCTIONS_DECL__ float atan2f(float a, float b)
{
  return __nv_atan2f(a, b);
}

__MATH_FUNCTIONS_DECL__ float atanf(float a)
{
  return __nv_atanf(a);
}

__MATH_FUNCTIONS_DECL__ float asinf(float a)
{
  return __nv_asinf(a);
}

__MATH_FUNCTIONS_DECL__ float acosf(float a)
{
  return __nv_acosf(a);
}

__MATH_FUNCTIONS_DECL__ float logf(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_logf(a);
#else /* __USE_FAST_MATH__ */
  return __nv_logf(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float log10f(float a)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_log10f(a);
#else /* __USE_FAST_MATH__ */
  return __nv_log10f(a);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float log1pf(float a)
{
  return __nv_log1pf(a);
}

__MATH_FUNCTIONS_DECL__ float acoshf(float a)
{
  return __nv_acoshf(a);
}

__MATH_FUNCTIONS_DECL__ float asinhf(float a)
{
  return __nv_asinhf(a);
}

__MATH_FUNCTIONS_DECL__ float atanhf(float a)
{
  return __nv_atanhf(a);
}

__MATH_FUNCTIONS_DECL__ float expm1f(float a)
{
  return __nv_expm1f(a);
}

__MATH_FUNCTIONS_DECL__ float hypotf(float a, float b)
{
  return __nv_hypotf(a, b);
}

__MATH_FUNCTIONS_DECL__ float rhypotf(float a, float b)
{
  return __nv_rhypotf(a, b);
}

__MATH_FUNCTIONS_DECL__ float norm3df(float a, float b, float c)
{
  return __nv_norm3df(a, b, c);
}

__MATH_FUNCTIONS_DECL__ float rnorm3df(float a, float b, float c)
{
  return __nv_rnorm3df(a, b, c);
}

__MATH_FUNCTIONS_DECL__ float norm4df(float a, float b, float c, float d)
{
  return __nv_norm4df(a, b, c, d);
}

__MATH_FUNCTIONS_DECL__ float cbrtf(float a)
{
  return __nv_cbrtf(a);
}

__MATH_FUNCTIONS_DECL__ float rcbrtf(float a)
{
  return __nv_rcbrtf(a);
}

__MATH_FUNCTIONS_DECL__ float j0f(float a)
{
  return __nv_j0f(a);
}

__MATH_FUNCTIONS_DECL__ float j1f(float a)
{
  return __nv_j1f(a);
}

__MATH_FUNCTIONS_DECL__ float y0f(float a)
{
  return __nv_y0f(a);
}

__MATH_FUNCTIONS_DECL__ float y1f(float a)
{
  return __nv_y1f(a);
}

__MATH_FUNCTIONS_DECL__ float ynf(int n, float a)
{
  return __nv_ynf(n, a);
}

__MATH_FUNCTIONS_DECL__ float jnf(int n, float a)
{
  return __nv_jnf(n, a);
}

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i0f(float a)
{
  return __nv_cyl_bessel_i0f(a);
}

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i1f(float a)
{
  return __nv_cyl_bessel_i1f(a);
}

__MATH_FUNCTIONS_DECL__ float erff(float a)
{
  return __nv_erff(a);
}

__MATH_FUNCTIONS_DECL__ float erfinvf(float a)
{
  return __nv_erfinvf(a);
}

__MATH_FUNCTIONS_DECL__ float erfcf(float a)
{
  return __nv_erfcf(a);
}

__MATH_FUNCTIONS_DECL__ float erfcxf(float a)
{
  return __nv_erfcxf(a);
}

__MATH_FUNCTIONS_DECL__ float erfcinvf(float a)
{
  return __nv_erfcinvf(a);
}

__MATH_FUNCTIONS_DECL__ float normcdfinvf(float a)
{
  return __nv_normcdfinvf(a);
}

__MATH_FUNCTIONS_DECL__ float normcdff(float a)
{
  return __nv_normcdff(a);
}

__MATH_FUNCTIONS_DECL__ float lgammaf(float a)
{
  return __nv_lgammaf(a);
}

__MATH_FUNCTIONS_DECL__ float ldexpf(float a, int b)
{
  return __nv_ldexpf(a, b);
}

__MATH_FUNCTIONS_DECL__ float scalbnf(float a, int b)
{
  return __nv_scalbnf(a, b);
}

__MATH_FUNCTIONS_DECL__ float scalblnf(float a, long int b)
{
  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return scalbnf(a, t);
}

__MATH_FUNCTIONS_DECL__ float frexpf(float a, int *b)
{
  return __nv_frexpf(a, b);
}

__MATH_FUNCTIONS_DECL__ float modff(float a, float *b)
{
  return __nv_modff(a, b);
}

__MATH_FUNCTIONS_DECL__ float fmodf(float a, float b)
{
  return __nv_fmodf(a, b);
}

__MATH_FUNCTIONS_DECL__ float remainderf(float a, float b)
{
  return __nv_remainderf(a, b);
}

__MATH_FUNCTIONS_DECL__ float remquof(float a, float b, int* quo)
{
  return __nv_remquof(a, b, quo);
}

__MATH_FUNCTIONS_DECL__ float fmaf(float a, float b, float c)
{
  return __nv_fmaf(a, b, c);
}

__MATH_FUNCTIONS_DECL__ float powif(float a, int b)
{
  return __nv_powif(a, b);
}

__MATH_FUNCTIONS_DECL__ double powi(double a, int b)
{
  return __nv_powi(a, b);
}

__MATH_FUNCTIONS_DECL__ float powf(float a, float b)
{
#if defined(__USE_FAST_MATH__)
  return __nv_fast_powf(a, b);
#else /* __USE_FAST_MATH__ */
  return __nv_powf(a, b);
#endif /* __USE_FAST_MATH__ */
}

__MATH_FUNCTIONS_DECL__ float tgammaf(float a)
{
  return __nv_tgammaf(a);
}

__MATH_FUNCTIONS_DECL__ float roundf(float a)
{
  return __nv_roundf(a);
}

__MATH_FUNCTIONS_DECL__ long long int llroundf(float a)
{
  return __nv_llroundf(a);
}

__MATH_FUNCTIONS_DECL__ long int lroundf(float a)
{
#if defined(__LP64__)
  return (long int)llroundf(a);
#else /* __LP64__ */
  return (long int)(roundf(a));
#endif /* __LP64__ */
}

__MATH_FUNCTIONS_DECL__ float fdimf(float a, float b)
{
  return __nv_fdimf(a, b);
}

__MATH_FUNCTIONS_DECL__ int ilogbf(float a)
{
  return __nv_ilogbf(a);
}

__MATH_FUNCTIONS_DECL__ float logbf(float a)
{
  return __nv_logbf(a);
}

#ifdef __QNX__
/* provide own versions of QNX builtins */
__MATH_FUNCTIONS_DECL__ float _FLog(float a, int tag)
{
  switch (tag) {
  case 0: return __logf(a);
  case 1: return __log10f(a);
  default: return __log2f(a);
  }
}
__MATH_FUNCTIONS_DECL__ float _FCosh (float a, float b)
{
  return coshf(a);
}
__MATH_FUNCTIONS_DECL__ float _FSinh (float a, float b)
{
  return sinhf(a);
}
__MATH_FUNCTIONS_DECL__ float _FSinx (float a, unsigned int tag, int c)
{
  if (tag == 1) return __cosf(a);
  else return __sinf(a);
}
__MATH_FUNCTIONS_DECL__ int _FDsign (float a)
{
  return __signbitf(a);
}
__MATH_FUNCTIONS_DECL__ int _Dsign (double a)
{
  return __signbit(a);
}
#endif /* QNX */

#undef __MATH_FUNCTIONS_DECL__

#else /* __CUDACC_RTC__ || __CUDABE__ */

/*******************************************************************************
*                                                                              *
* ONLY FOR HOST CODE! NOT FOR DEVICE EXECUTION                                 *
*                                                                              *
*******************************************************************************/

#include <crt/func_macro.h>

#if defined(_WIN32)

#pragma warning(disable : 4211)

#endif /* _WIN32 */

#if defined(_WIN32) || defined(__APPLE__) || defined (__ANDROID__) || defined(__QNX__)

__func__(int __isnan(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 > 0xffe0000000000000ull;
}

#endif /* _WIN32 || __APPLE__ || __ANDROID__ || __QNX__ */

#if defined(_WIN32) || defined(__APPLE__) || defined(__QNX__)

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR DOUBLE ROUTINES FOR WINDOWS & APPLE PLATFORMS        *
*                                                                              *
*******************************************************************************/

__func__(double exp10(double a))
{
  return pow(10.0, a);
}

__func__(float exp10f(float a))
{
    return (float)exp10((double)a);
}

__func__(void sincos(double a, double *sptr, double *cptr))
{
  *sptr = sin(a);
  *cptr = cos(a);
}

__func__(void sincosf(float a, float *sptr, float *cptr))
{
  double s, c;

  sincos((double)a, &s, &c);
  *sptr = (float)s;
  *cptr = (float)c;
}

__func__(int __isinf(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 == 0xffe0000000000000ull;
}

#endif /* _WIN32 || __APPLE__ */

#if defined(_WIN32) || defined (__ANDROID__)

#if _MSC_VER < 1800
__func__(double log2(double a))
{
  return log(a) * 1.44269504088896340;
}
#endif /* _MSC_VER < 1800 */

#endif /* _WIN32 || __ANDROID__ */

#if defined(_WIN32)

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR DOUBLE ROUTINES FOR WINDOWS PLATFORM                 *
*                                                                              *
*******************************************************************************/

__func__(int __signbit(double a))
{
  volatile union {
    double               d;
    signed long long int l;
  } cvt;

  cvt.d = a;
  return cvt.l < 0ll;
}

#if _MSC_VER < 1800
__func__(double copysign(double a, double b))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvta, cvtb;

  cvta.d = a;
  cvtb.d = b;
  cvta.l = (cvta.l & 0x7fffffffffffffffULL) | (cvtb.l & 0x8000000000000000ULL);
  return cvta.d;
}
#endif /* MSC_VER < 1800 */

__func__(int __finite(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;
  return cvt.l << 1 < 0xffe0000000000000ull;
}

#if _MSC_VER < 1800
__func__(double fmax(double a, double b))
{
  if (__isnan(a) && __isnan(b)) return a + b;
  if (__isnan(a)) return b;
  if (__isnan(b)) return a;
  if ((a == 0.0) && (b == 0.0) && __signbit(b)) return a;
  return a > b ? a : b;
}

__func__(double fmin(double a, double b))
{
  if (__isnan(a) && __isnan(b)) return a + b;
  if (__isnan(a)) return b;
  if (__isnan(b)) return a;
  if ((a == 0.0) && (b == 0.0) && __signbit(a)) return a;
  return a < b ? a : b;
}

__func__(double trunc(double a))
{
  return a < 0.0 ? ceil(a) : floor(a);
}

__func__(double round(double a))
{
  double fa = fabs(a);

  if (fa > CUDART_TWO_TO_52) {
    return a;
  } else {
    double u = floor(fa + 0.5);
    if (fa < 0.5) u = 0;
    u = copysign (u, a);
    return u;
  }
}

__func__(long int lround(double a))
{
  return (long int)round(a);
}

__func__(long long int llround(double a))
{
  return (long long int)round(a);
}

__func__(double rint(double a))
{
  double fa = fabs(a);
  double u = CUDART_TWO_TO_52 + fa;
  if (fa >= CUDART_TWO_TO_52) {
    u = a;
  } else {
    u = u - CUDART_TWO_TO_52;
    u = copysign (u, a);
  }
  return u;  
}

__func__(double nearbyint(double a))
{
  return rint(a);
}

__func__(long int lrint(double a))
{
  return (long int)rint(a);
}

__func__(long long int llrint(double a))
{
  return (long long int)rint(a);
}

__func__(double fdim(double a, double b))
{
  if (a > b) {
    return (a - b);
  } else if (a <= b) {
    return 0.0;
  } else if (__isnan(a)) {
    return a;
  } else {
    return b;
  }
}

__func__(double scalbn(double a, int b))
{
  return ldexp(a, b);
}

__func__(double scalbln(double a, long int b))
{
  int t;

  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return scalbn(a, t);
}

__func__(double exp2(double a))
{
  return pow(2.0, a);
}

/*  
 * The following is based on: David Goldberg, "What every computer scientist 
 * should know about floating-point arithmetic", ACM Computing Surveys, Volume 
 * 23, Issue 1, March 1991.
 */
__func__(double log1p(double a))
{
  volatile double u, m;

  u = 1.0 + a;
  if (u == 1.0) {
    /* a very close to zero */
    u = a;
  } else {
    m = u - 1.0;
    u = log(u);
    if (a < 1.0) {
      /* a somewhat close to zero */
      u = a * u;
      u = u / m;
    }
  }
  return u;
}

/*
 * This code based on: http://www.cs.berkeley.edu/~wkahan/Math128/Sumnfp.pdf
 */
__func__(double expm1(double a))
{
  volatile double u, m;

  u = exp(a);
  m = u - 1.0;
  if (m == 0.0) {
    /* a very close zero */
    m = a;
  } 
  else if (fabs(a) < 1.0) {
    /* a somewhat close zero */
    u = log(u);
    m = m * a;
    m = m / u;
  }
  return m;
}

__func__(double cbrt(double a))
{
  double s, t;

  if (a == 0.0 || __isinf(a)) {
    return a;
  } 
  s = fabs(a);
  t = exp2(CUDART_THIRD * log2(s));           /* initial approximation */
  t = t - (t - (s / (t * t))) * CUDART_THIRD; /* refine approximation */
  t = copysign(t, a);
  return t;
}

__func__(double acosh(double a))
{
  double s, t;

  t = a - 1.0;
  if (t == a) {
    return log(2.0) + log(a);
  } else {
    s = a + 1.0;
    t = t + sqrt(s * t);
    return log1p(t);
  }
}

__func__(double asinh(double a))
{
  double fa, oofa, t;

  fa = fabs(a);
  if (fa > 1e18) {
    t = log(2.0) + log(fa);
  } else {
    oofa = 1.0 / fa;
    t = fa + fa / (oofa + sqrt(1.0 + oofa * oofa));
    t = log1p(t);
  }
  t = copysign(t, a);
  return t;
}

__func__(double atanh(double a))
{
  double fa, t;

  if (__isnan(a)) {
    return a + a;
  }
  fa = fabs(a);
  t = (2.0 * fa) / (1.0 - fa);
  t = 0.5 * log1p(t);
  if (__isnan(t) || !__signbit(a)) {
    return t;
  }
  return -t;
}

__func__(int ilogb(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } x;
  unsigned long long int i;
  int expo = -1022;

  if (__isnan(a)) return -__cuda_INT_MAX-1;
  if (__isinf(a)) return __cuda_INT_MAX;
  x.d = a;
  i = x.l & 0x7fffffffffffffffull;
  if (i == 0) return -__cuda_INT_MAX-1;
  if (i >= 0x0010000000000000ull) {
    return (int)(((i >> 52) & 0x7ff) - 1023);
  }
  while (i < 0x0010000000000000ull) {
    expo--;
    i <<= 1;
  }
  return expo;
}

__func__(double logb(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } x;
  unsigned long long int i;
  int expo = -1022;

  if (__isnan(a)) return a + a;
  if (__isinf(a)) return fabs(a);
  x.d = a;
  i = x.l & 0x7fffffffffffffffull;
  if (i == 0) return -1.0/fabs(a);
  if (i >= 0x0010000000000000ull) {
    return (double)((int)((i >> 52) & 0x7ff) - 1023);
  }
  while (i < 0x0010000000000000ull) {
    expo--;
    i <<= 1;
  }
  return (double)expo;
}

__func__(double remquo(double a, double b, int *quo))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;
  int rem1 = 1; /* do FPREM1, a.k.a IEEE remainder */
  int expo_a;
  int expo_b;
  unsigned long long mant_a;
  unsigned long long mant_b;
  unsigned long long mant_c;
  unsigned long long temp;
  int sign_a;
  int sign_b;
  int sign_c;
  int expo_c;
  int expodiff;
  int quot = 0;                 /* initialize quotient */
  int l;
  int iter;

  cvt.d = a;
  mant_a = (cvt.l << 11) | 0x8000000000000000ULL;
  expo_a = (int)((cvt.l >> 52) & 0x7ff) - 1023;
  sign_a = (int)(cvt.l >> 63);

  cvt.d = b;
  mant_b = (cvt.l << 11) | 0x8000000000000000ULL;
  expo_b = (int)((cvt.l >> 52) & 0x7ff) - 1023;
  sign_b = (int)(cvt.l >> 63);

  sign_c = sign_a;  /* remainder has sign of dividend */
  expo_c = expo_a;  /* default */
      
  /* handled NaNs and infinities */
  if (__isnan(a) || __isnan(b)) {
    *quo = quot;
    return a + b;
  }
  if (__isinf(a) || (b == 0.0)) {
    *quo = quot;
    cvt.l = 0xfff8000000000000ULL;
    return cvt.d;
  }
  if ((a == 0.0) || (__isinf(b))) {
    *quo = quot;
    return a;
  }
  /* normalize denormals */
  if (expo_a < -1022) {
    mant_a = mant_a + mant_a;
    while (mant_a < 0x8000000000000000ULL) {
      mant_a = mant_a + mant_a;
      expo_a--;
    }
  } 
  if (expo_b < -1022) {
    mant_b = mant_b + mant_b;
    while (mant_b < 0x8000000000000000ULL) {
      mant_b = mant_b + mant_b;
      expo_b--;
    }
  }
  expodiff = expo_a - expo_b;
  /* clamp iterations if exponent difference negative */
  if (expodiff < 0) {
    iter = -1;
  } else {
    iter = expodiff;
  }
  /* Shift dividend and divisor right by one bit to prevent overflow
     during the division algorithm.
   */
  mant_a = mant_a >> 1;
  mant_b = mant_b >> 1;
  expo_c = expo_a - iter; /* default exponent of result   */

  /* Use binary longhand division (restoring) */
  for (l = 0; l < (iter + 1); l++) {
    mant_a = mant_a - mant_b;
    if (mant_a & 0x8000000000000000ULL) {
      mant_a = mant_a + mant_b;
      quot = quot + quot;
    } else {
      quot = quot + quot + 1;
    }
    mant_a = mant_a + mant_a;
  }

  /* Save current remainder */
  mant_c = mant_a;
  /* If remainder's mantissa is all zeroes, final result is zero. */
  if (mant_c == 0) {
    quot = quot & 7;
    *quo = (sign_a ^ sign_b) ? -quot : quot;
    cvt.l = (unsigned long long int)sign_c << 63;
    return cvt.d;
  }
  /* Normalize result */
  while (!(mant_c & 0x8000000000000000ULL)) {
    mant_c = mant_c + mant_c;
    expo_c--;
  }
  /* For IEEE remainder (quotient rounded to nearest-even we might need to 
     do a final subtraction of the divisor from the remainder.
  */
  if (rem1 && ((expodiff+1) >= 0)) {
    temp = mant_a - mant_b;
    /* round quotient to nearest even */
    if (((temp != 0ULL) && (!(temp & 0x8000000000000000ULL))) ||
        ((temp == 0ULL) && (quot & 1))) {
      mant_a = mant_a >> 1;
      quot++;
      /* Since the divisor is greater than the remainder, the result will
         have opposite sign of the dividend. To avoid a negative mantissa
         when subtracting the divisor from remainder, reverse subtraction
      */
      sign_c = 1 ^ sign_c;
      expo_c = expo_a - iter + 1;
      mant_c = mant_b - mant_a;
      /* normalize result */
      while (!(mant_c & 0x8000000000000000ULL)) {
        mant_c = mant_c + mant_c;
        expo_c--;
      }
    }
  }
  /* package up result */
  if (expo_c >= -1022) { /* normal */
    mant_c = ((mant_c >> 11) + 
              ((((unsigned long long)sign_c) << 63) +
               (((unsigned long long)(expo_c + 1022)) << 52)));
  } else { /* denormal */
    mant_c = ((((unsigned long long)sign_c) << 63) + 
              (mant_c >> (11 - expo_c - 1022)));
  }
  quot = quot & 7; /* mask quotient down to least significant three bits */
  *quo = (sign_a ^ sign_b) ? -quot : quot;
  cvt.l = mant_c;
  return cvt.d;
}

__func__(double remainder(double a, double b))
{
  int quo;
  return remquo (a, b, &quo);
}

__func__(double fma (double a, double b, double c))
{
  volatile union {
    struct {
      unsigned int lo;
      unsigned int hi;
    } part;
    double d;
  } xx, yy, zz, ww;
  unsigned int s, t, u, prod0, prod1, prod2, prod3, expo_x, expo_y, expo_z;
  
  xx.d = a;
  yy.d = b;
  zz.d = c;

  expo_z = 0x7FF;
  t =  xx.part.hi >> 20;
  expo_x = expo_z & t;
  expo_x = expo_x - 1;    /* expo(x) - 1 */
  t =  yy.part.hi >> 20;
  expo_y = expo_z & t;
  expo_y = expo_y - 1;    /* expo(y) - 1 */
  t =  zz.part.hi >> 20;
  expo_z = expo_z & t;
  expo_z = expo_z - 1;    /* expo(z) - 1 */

  if (!((expo_x <= 0x7FD) &&
        (expo_y <= 0x7FD) &&
        (expo_z <= 0x7FD))) {
    
    /* fma (nan, y, z) --> nan
       fma (x, nan, z) --> nan
       fma (x, y, nan) --> nan 
    */
    if (((yy.part.hi << 1) | (yy.part.lo != 0)) > 0xffe00000) {
      yy.part.hi |= 0x00080000;
      return yy.d;
    }
    if (((zz.part.hi << 1) | (zz.part.lo != 0)) > 0xffe00000) {
      zz.part.hi |= 0x00080000;
      return zz.d;
    }
    if (((xx.part.hi << 1) | (xx.part.lo != 0)) > 0xffe00000) {
      xx.part.hi |= 0x00080000;
      return xx.d;
    }
    
    /* fma (0, inf, z) --> INDEFINITE
       fma (inf, 0, z) --> INDEFINITE
       fma (-inf,+y,+inf) --> INDEFINITE
       fma (+x,-inf,+inf) --> INDEFINITE
       fma (+inf,-y,+inf) --> INDEFINITE
       fma (-x,+inf,+inf) --> INDEFINITE
       fma (-inf,-y,-inf) --> INDEFINITE
       fma (-x,-inf,-inf) --> INDEFINITE
       fma (+inf,+y,-inf) --> INDEFINITE
       fma (+x,+inf,-inf) --> INDEFINITE
    */
    if (((((xx.part.hi << 1) | xx.part.lo) == 0) && 
         (((yy.part.hi << 1) | (yy.part.lo != 0)) == 0xffe00000)) ||
        ((((yy.part.hi << 1) | yy.part.lo) == 0) && 
         (((xx.part.hi << 1) | (xx.part.lo != 0)) == 0xffe00000))) {
      xx.part.hi = 0xfff80000;
      xx.part.lo = 0x00000000;
      return xx.d;
    }
    if (((zz.part.hi << 1) | (zz.part.lo != 0)) == 0xffe00000) {
      if ((((yy.part.hi << 1) | (yy.part.lo != 0)) == 0xffe00000) ||
          (((xx.part.hi << 1) | (xx.part.lo != 0)) == 0xffe00000)) {
        if ((int)(xx.part.hi ^ yy.part.hi ^ zz.part.hi) < 0) {
          xx.part.hi = 0xfff80000;
          xx.part.lo = 0x00000000;
          return xx.d;
        }
      }
    }
    /* fma (inf, y, z) --> inf
       fma (x, inf, z) --> inf
       fma (x, y, inf) --> inf
    */
    if (((xx.part.hi << 1) | (xx.part.lo != 0)) == 0xffe00000) {
      xx.part.hi = xx.part.hi ^ (yy.part.hi & 0x80000000);
      return xx.d;
    }
    if (((yy.part.hi << 1) | (yy.part.lo != 0)) == 0xffe00000) {
      yy.part.hi = yy.part.hi ^ (xx.part.hi & 0x80000000);
      return yy.d;
    }
    if (((zz.part.hi << 1) | (zz.part.lo != 0)) == 0xffe00000) {
      return zz.d;
    }
    /* fma (+0, -y, -0) --> -0
       fma (-0, +y, -0) --> -0
       fma (+x, -0, -0) --> -0
       fma (-x, +0, -0) --> -0
    */
    if ((zz.part.hi == 0x80000000) && (zz.part.lo == 0)) {
      if ((((xx.part.hi << 1) | xx.part.lo) == 0) ||
          (((yy.part.hi << 1) | yy.part.lo) == 0)) {
        if ((int)(xx.part.hi ^ yy.part.hi) < 0) {
          return zz.d;
        }
      }
    }
    /* fma (0, y, 0) --> +0  (-0 if round down and signs of addend differ)
       fma (x, 0, 0) --> +0  (-0 if round down and signs of addend differ)
    */
    if ((((zz.part.hi << 1) | zz.part.lo) == 0) &&
        ((((xx.part.hi << 1) | xx.part.lo) == 0) ||
         (((yy.part.hi << 1) | yy.part.lo) == 0))) {
      zz.part.hi &= 0x7fffffff;
      return zz.d;
    }
    
    /* fma (0, y, z) --> z
       fma (x, 0, z) --> z
    */
    if ((((xx.part.hi << 1) | xx.part.lo) == 0) ||
        (((yy.part.hi << 1) | yy.part.lo) == 0)) {
      return zz.d;
    }
    
    if (expo_x == 0xffffffff) {
      expo_x++;
      t = xx.part.hi & 0x80000000;
      s = xx.part.lo >> 21;
      xx.part.lo = xx.part.lo << 11;
      xx.part.hi = xx.part.hi << 11;
      xx.part.hi = xx.part.hi | s;
      if (!xx.part.hi) {
        xx.part.hi = xx.part.lo;
        xx.part.lo = 0;
        expo_x -= 32;
      }
      while ((int)xx.part.hi > 0) {
        s = xx.part.lo >> 31;
        xx.part.lo = xx.part.lo + xx.part.lo;
        xx.part.hi = xx.part.hi + xx.part.hi;
        xx.part.hi = xx.part.hi | s;
        expo_x--;
      }
      xx.part.lo = (xx.part.lo >> 11);
      xx.part.lo |= (xx.part.hi << 21);
      xx.part.hi = (xx.part.hi >> 11) | t;
    }
    if (expo_y == 0xffffffff) {
      expo_y++;
      t = yy.part.hi & 0x80000000;
      s = yy.part.lo >> 21;
      yy.part.lo = yy.part.lo << 11;
      yy.part.hi = yy.part.hi << 11;
      yy.part.hi = yy.part.hi | s;
      if (!yy.part.hi) {
        yy.part.hi = yy.part.lo;
        yy.part.lo = 0;
        expo_y -= 32;
      }
      while ((int)yy.part.hi > 0) {
        s = yy.part.lo >> 31;
        yy.part.lo = yy.part.lo + yy.part.lo;
        yy.part.hi = yy.part.hi + yy.part.hi;
        yy.part.hi = yy.part.hi | s;
        expo_y--;
      }
      yy.part.lo = (yy.part.lo >> 11);
      yy.part.lo |= (yy.part.hi << 21);
      yy.part.hi = (yy.part.hi >> 11) | t;
    }
    if (expo_z == 0xffffffff) {
      expo_z++;
      t = zz.part.hi & 0x80000000;
      s = zz.part.lo >> 21;
      zz.part.lo = zz.part.lo << 11;
      zz.part.hi = zz.part.hi << 11;
      zz.part.hi = zz.part.hi | s;
      if (!zz.part.hi) {
        zz.part.hi = zz.part.lo;
        zz.part.lo = 0;
        expo_z -= 32;
      }
      while ((int)zz.part.hi > 0) {
        s = zz.part.lo >> 31;
        zz.part.lo = zz.part.lo + zz.part.lo;
        zz.part.hi = zz.part.hi + zz.part.hi;
        zz.part.hi = zz.part.hi | s;
        expo_z--;
      }
      zz.part.lo = (zz.part.lo >> 11);
      zz.part.lo |= (zz.part.hi << 21);
      zz.part.hi = (zz.part.hi >> 11) | t;
    }
  }
  
  expo_x = expo_x + expo_y;
  expo_y = xx.part.hi ^ yy.part.hi;
  t = xx.part.lo >> 21;
  xx.part.lo = xx.part.lo << 11;
  xx.part.hi = xx.part.hi << 11;
  xx.part.hi = xx.part.hi | t;
  yy.part.hi = yy.part.hi & 0x000fffff;
  xx.part.hi = xx.part.hi | 0x80000000; /* set mantissa hidden bit */
  yy.part.hi = yy.part.hi | 0x00100000; /* set mantissa hidden bit */

  prod0 = xx.part.lo * yy.part.lo;
  prod1 =(unsigned)(((unsigned long long)xx.part.lo*(unsigned long long)yy.part.lo)>>32);
  prod2 = xx.part.hi * yy.part.lo;
  prod3 = xx.part.lo * yy.part.hi;
  prod1 += prod2;
  t = (unsigned)(prod1 < prod2);
  prod1 += prod3;
  t += prod1 < prod3;
  prod2 =(unsigned)(((unsigned long long)xx.part.hi*(unsigned long long)yy.part.lo)>>32);
  prod3 =(unsigned)(((unsigned long long)xx.part.lo*(unsigned long long)yy.part.hi)>>32);
  prod2 += prod3;
  s = (unsigned)(prod2 < prod3);
  prod3 = xx.part.hi * yy.part.hi;
  prod2 += prod3;
  s += prod2 < prod3;
  prod2 += t;
  s += prod2 < t;
  prod3 =(unsigned)(((unsigned long long)xx.part.hi*(unsigned long long)yy.part.hi)>>32);
  prod3 = prod3 + s;
  
  yy.part.lo = prod0;                 /* mantissa */
  yy.part.hi = prod1;                 /* mantissa */
  xx.part.lo = prod2;                 /* mantissa */
  xx.part.hi = prod3;                 /* mantissa */
  expo_x = expo_x - (1023 - 2);  /* expo-1 */
  expo_y = expo_y & 0x80000000;  /* sign */

  if (xx.part.hi < 0x00100000) {
    s = xx.part.lo >> 31;
    s = (xx.part.hi << 1) + s;
    xx.part.hi = s;
    s = yy.part.hi >> 31;
    s = (xx.part.lo << 1) + s;
    xx.part.lo = s;
    s = yy.part.lo >> 31;
    s = (yy.part.hi << 1) + s;
    yy.part.hi = s;
    s = yy.part.lo << 1;
    yy.part.lo = s;
    expo_x--;
  }

  t = 0;
  if (((zz.part.hi << 1) | zz.part.lo) != 0) { /* z is not zero */
    
    s = zz.part.hi & 0x80000000;
    
    zz.part.hi &= 0x000fffff;
    zz.part.hi |= 0x00100000;
    ww.part.hi = 0;
    ww.part.lo = 0;
    
    /* compare and swap. put augend into xx:yy */
    if ((int)expo_z > (int)expo_x) {
      t = expo_z;
      expo_z = expo_x;
      expo_x = t;
      t = zz.part.hi;
      zz.part.hi = xx.part.hi;
      xx.part.hi = t;
      t = zz.part.lo;
      zz.part.lo = xx.part.lo;
      xx.part.lo = t;
      t = ww.part.hi;
      ww.part.hi = yy.part.hi;
      yy.part.hi = t;
      t = ww.part.lo;
      ww.part.lo = yy.part.lo;
      yy.part.lo = t;
      t = expo_y;
      expo_y = s;
      s = t;
    }
    
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 107) {
      /* denormalize addend */
      t = 0;
      while (expo_z >= 32) {
        t     = ww.part.lo | (t != 0);
        ww.part.lo = ww.part.hi;
        ww.part.hi = zz.part.lo;
        zz.part.lo = zz.part.hi;
        zz.part.hi = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        t     = (t     >> expo_z) | (ww.part.lo << (32 - expo_z)) | 
                ((t << (32 - expo_z)) != 0);
        ww.part.lo = (ww.part.lo >> expo_z) | (ww.part.hi << (32 - expo_z));
        ww.part.hi = (ww.part.hi >> expo_z) | (zz.part.lo << (32 - expo_z));
        zz.part.lo = (zz.part.lo >> expo_z) | (zz.part.hi << (32 - expo_z));
        zz.part.hi = (zz.part.hi >> expo_z);
      }
    } else {
      t = 1;
      ww.part.lo = 0;
      ww.part.hi = 0;
      zz.part.lo = 0;
      zz.part.hi = 0;
    }
    if ((int)u < 0) {
      /* signs differ, effective subtraction */
      t = (unsigned)(-(int)t);
      s = (unsigned)(t != 0);
      u = yy.part.lo - s;
      s = (unsigned)(u > yy.part.lo);
      yy.part.lo = u - ww.part.lo;
      s += yy.part.lo > u;
      u = yy.part.hi - s;
      s = (unsigned)(u > yy.part.hi);
      yy.part.hi = u - ww.part.hi;
      s += yy.part.hi > u;
      u = xx.part.lo - s;
      s = (unsigned)(u > xx.part.lo);
      xx.part.lo = u - zz.part.lo;
      s += xx.part.lo > u;
      xx.part.hi = (xx.part.hi - zz.part.hi) - s;
      if (!(xx.part.hi | xx.part.lo | yy.part.hi | yy.part.lo | t)) {
        /* complete cancelation, return 0 */
        return xx.d;
      }
      if ((int)xx.part.hi < 0) {
        /* Oops, augend had smaller mantissa. Negate mantissa and flip
           sign of result
        */
        t = ~t;
        yy.part.lo = ~yy.part.lo;
        yy.part.hi = ~yy.part.hi;
        xx.part.lo = ~xx.part.lo;
        xx.part.hi = ~xx.part.hi;
        if (++t == 0) {
          if (++yy.part.lo == 0) {
            if (++yy.part.hi == 0) {
              if (++xx.part.lo == 0) {
              ++xx.part.hi;
              }
            }
          }
        }
        expo_y ^= 0x80000000;
      }
        
      /* normalize mantissa, if necessary */
      while (!(xx.part.hi & 0x00100000)) {
        xx.part.hi = (xx.part.hi << 1) | (xx.part.lo >> 31);
        xx.part.lo = (xx.part.lo << 1) | (yy.part.hi >> 31);
        yy.part.hi = (yy.part.hi << 1) | (yy.part.lo >> 31);
        yy.part.lo = (yy.part.lo << 1);
        expo_x--;
      }
    } else {
      /* signs are the same, effective addition */
      yy.part.lo = yy.part.lo + ww.part.lo;
      s = (unsigned)(yy.part.lo < ww.part.lo);
      yy.part.hi = yy.part.hi + s;
      u = (unsigned)(yy.part.hi < s);
      yy.part.hi = yy.part.hi + ww.part.hi;
      u += yy.part.hi < ww.part.hi;
      xx.part.lo = xx.part.lo + u;
      s = (unsigned)(xx.part.lo < u);
      xx.part.lo = xx.part.lo + zz.part.lo;
      s += xx.part.lo < zz.part.lo;
      xx.part.hi = xx.part.hi + zz.part.hi + s;
      if (xx.part.hi & 0x00200000) {
        t = t | (yy.part.lo << 31);
        yy.part.lo = (yy.part.lo >> 1) | (yy.part.hi << 31);
        yy.part.hi = (yy.part.hi >> 1) | (xx.part.lo << 31);
        xx.part.lo = (xx.part.lo >> 1) | (xx.part.hi << 31);
        xx.part.hi = ((xx.part.hi & 0x80000000) | (xx.part.hi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  t = yy.part.lo | (t != 0);
  t = yy.part.hi | (t != 0);
        
  xx.part.hi |= expo_y; /* or in sign bit */
  if (expo_x <= 0x7FD) {
    /* normal */
    xx.part.hi = xx.part.hi & ~0x00100000; /* lop off integer bit */
    s = xx.part.lo & 1; /* mantissa lsb */
    u = xx.part.lo;
    xx.part.lo += (t == 0x80000000) ? s : (t >> 31);
    xx.part.hi += (u > xx.part.lo);
    xx.part.hi += ((expo_x + 1) << 20);
    return xx.d;
  } else if ((int)expo_x >= 2046) {      
    /* overflow */
    xx.part.hi = (xx.part.hi & 0x80000000) | 0x7ff00000;
    xx.part.lo = 0;
    return xx.d;
  }
  /* subnormal */
  expo_x = (unsigned)(-(int)expo_x);
  if (expo_x > 54) {
    xx.part.hi = xx.part.hi & 0x80000000;
    xx.part.lo = 0;
    return xx.d;
  }  
  yy.part.hi = xx.part.hi &  0x80000000;   /* save sign bit */
  xx.part.hi = xx.part.hi & ~0xffe00000;
  if (expo_x >= 32) {
    t = xx.part.lo | (t != 0);
    xx.part.lo = xx.part.hi;
    xx.part.hi = 0;
    expo_x -= 32;
  }
  if (expo_x) {
    t     = (t     >> expo_x) | (xx.part.lo << (32 - expo_x)) | (t != 0);
    xx.part.lo = (xx.part.lo >> expo_x) | (xx.part.hi << (32 - expo_x));
    xx.part.hi = (xx.part.hi >> expo_x);
  }
  expo_x = xx.part.lo & 1; 
  u = xx.part.lo;
  xx.part.lo += (t == 0x80000000) ? expo_x : (t >> 31);
  xx.part.hi += (u > xx.part.lo);
  xx.part.hi |= yy.part.hi;
  return xx.d;
}

__func__(double nextafter(double a, double b))
{
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;
  unsigned long long int ia;
  unsigned long long int ib;
  cvt.d = a;
  ia = cvt.l;
  cvt.d = b;
  ib = cvt.l;
  if (__isnan(a) || __isnan(b)) return a + b; /* NaN */
  if (((ia | ib) << 1) == 0ULL) return b;
  if (a == 0.0) {
    return copysign (4.9406564584124654e-324, b); /* crossover */
  }
  if ((a < b) && (a < 0.0)) ia--;
  if ((a < b) && (a > 0.0)) ia++;
  if ((a > b) && (a < 0.0)) ia++;
  if ((a > b) && (a > 0.0)) ia--;
  cvt.l = ia;
  return cvt.d;
}

__func__(double erf(double a))
{
  double t, r, q;

  t = fabs(a);
  if (t >= 1.0) {
    r =        -1.28836351230756500E-019;
    r = r * t + 1.30597472161093370E-017;
    r = r * t - 6.33924401259620500E-016;
    r = r * t + 1.96231865908940140E-014;
    r = r * t - 4.35272243559990750E-013;
    r = r * t + 7.37083927929352150E-012;
    r = r * t - 9.91402142550461630E-011;
    r = r * t + 1.08817017167760820E-009;
    r = r * t - 9.93918713097634620E-009;
    r = r * t + 7.66739923255145500E-008;
    r = r * t - 5.05440278302806720E-007;
    r = r * t + 2.87474157099000620E-006;
    r = r * t - 1.42246725399722510E-005;
    r = r * t + 6.16994555079419460E-005;
    r = r * t - 2.36305221938908790E-004;
    r = r * t + 8.05032844055371070E-004;
    r = r * t - 2.45833366629108140E-003;
    r = r * t + 6.78340988296706120E-003;
    r = r * t - 1.70509103597554640E-002;
    r = r * t + 3.93322852515666300E-002;
    r = r * t - 8.37271292613764040E-002;
    r = r * t + 1.64870423707623280E-001;
    r = r * t - 2.99729521787681470E-001;
    r = r * t + 4.99394435612628580E-001;
    r = r * t - 7.52014596480123030E-001;
    r = r * t + 9.99933138314926250E-001;
    r = r * t - 1.12836725321102670E+000;
    r = r * t + 9.99998988715182450E-001;
    q = exp (-t * t);
    r = 1.0 - r * q;
    if (t >= 6.5) {
      r = 1.0;
    }    
    a = copysign (r, a);
  } else {
    q = a * a;
    r =        -7.77946848895991420E-010;
    r = r * q + 1.37109803980285950E-008;
    r = r * q - 1.62063137584932240E-007;
    r = r * q + 1.64471315712790040E-006;
    r = r * q - 1.49247123020098620E-005;
    r = r * q + 1.20552935769006260E-004;
    r = r * q - 8.54832592931448980E-004;
    r = r * q + 5.22397760611847340E-003;
    r = r * q - 2.68661706431114690E-002;
    r = r * q + 1.12837916709441850E-001;
    r = r * q - 3.76126389031835210E-001;
    r = r * q + 1.12837916709551260E+000;
    a = r * a;
  }
  return a;
}

__func__(double erfc(double a))
{
  double p, q, h, l;

  if (a < 0.75) {
    return 1.0 - erf(a);
  } 
  if (a > 27.3) {
    return 0.0;
  }
  if (a < 5.0) {
    double t;
    t = 1.0 / a;
    p =         1.9759923722227928E-008;
    p = p * t - 1.0000002670474897E+000;
    p = p * t - 7.4935303236347828E-001;
    p = p * t - 1.5648136328071860E-001;
    p = p * t + 1.2871196242447239E-001;
    p = p * t + 1.1126459974811195E-001;
    p = p * t + 4.0678642255914332E-002;
    p = p * t + 7.9915414156678296E-003;
    p = p * t + 7.1458332107840234E-004;
    q =     t + 2.7493547525030619E+000;
    q = q * t + 3.3984254815725423E+000;
    q = q * t + 2.4635304979947761E+000;
    q = q * t + 1.1405284734691286E+000;
    q = q * t + 3.4130157606195649E-001;
    q = q * t + 6.2250967676044953E-002;
    q = q * t + 5.5661370941268700E-003;
    q = q * t + 1.0575248365468671E-009;
    p = p / q;
    p = p * t;
    h = ((int)(a * 16.0)) * 0.0625;
    l = (a - h) * (a + h);
    q = exp(-h * h) * exp(-l);
    q = q * 0.5;
    p = p * q + q;
    p = p * t;
  } else {
    double ooa, ooasq;

    ooa = 1.0 / a;
    ooasq = ooa * ooa;
    p =            -4.0025406686930527E+005;
    p = p * ooasq + 1.4420582543942123E+005;
    p = p * ooasq - 2.7664185780951841E+004;
    p = p * ooasq + 4.1144611644767283E+003;
    p = p * ooasq - 5.8706000519209351E+002;
    p = p * ooasq + 9.1490086446323375E+001;
    p = p * ooasq - 1.6659491387740221E+001;
    p = p * ooasq + 3.7024804085481784E+000;
    p = p * ooasq - 1.0578553994424316E+000;
    p = p * ooasq + 4.2314218745087778E-001;
    p = p * ooasq - 2.8209479177354962E-001;
    p = p * ooasq + 5.6418958354775606E-001;
    h = a * a;
    h = ((int)(a * 16.0)) * 0.0625;
    l = (a - h) * (a + h);
    q = exp(-h * h) * exp(-l);
    p = p * ooa;
    p = p * q;
  }
  return p;
}

__func__(double lgamma(double a))
{
  double s;
  double t;
  double i;
  double fa;
  double sum;
  long long int quot;
  if (__isnan(a) || __isinf(a)) {
    return a * a;
  }
  fa = fabs(a);
  if (fa >= 3.0) {
    if (fa >= 8.0) {
      /* Stirling approximation; coefficients from Hart et al, "Computer 
       * Approximations", Wiley 1968. Approximation 5404. 
       */
      s = 1.0 / fa;
      t = s * s;
      sum =          -0.1633436431e-2;
      sum = sum * t + 0.83645878922e-3;
      sum = sum * t - 0.5951896861197e-3;
      sum = sum * t + 0.793650576493454e-3;
      sum = sum * t - 0.277777777735865004e-2;
      sum = sum * t + 0.833333333333331018375e-1;
      sum = sum * s + 0.918938533204672;
      s = 0.5 * log (fa);
      t = fa - 0.5;
      s = s * t;
      t = s - fa;
      s = s + sum;
      t = t + s;
    } else {
      i = fa - 3.0;
      s =        -4.02412642744125560E+003;
      s = s * i - 2.97693796998962000E+005;
      s = s * i - 6.38367087682528790E+006;
      s = s * i - 5.57807214576539320E+007;
      s = s * i - 2.24585140671479230E+008;
      s = s * i - 4.70690608529125090E+008;
      s = s * i - 7.62587065363263010E+008;
      s = s * i - 9.71405112477113250E+008;
      t =     i - 1.02277248359873170E+003;
      t = t * i - 1.34815350617954480E+005;
      t = t * i - 4.64321188814343610E+006;
      t = t * i - 6.48011106025542540E+007;
      t = t * i - 4.19763847787431360E+008;
      t = t * i - 1.25629926018000720E+009;
      t = t * i - 1.40144133846491690E+009;
      t = s / t;
      t = t + i;
    }
  } else if (fa >= 1.5) {
    i = fa - 2.0;
    t =         9.84839283076310610E-009;
    t = t * i - 6.69743850483466500E-008;
    t = t * i + 2.16565148880011450E-007;
    t = t * i - 4.86170275781575260E-007;
    t = t * i + 9.77962097401114400E-007;
    t = t * i - 2.03041287574791810E-006;
    t = t * i + 4.36119725805364580E-006;
    t = t * i - 9.43829310866446590E-006;
    t = t * i + 2.05106878496644220E-005;
    t = t * i - 4.49271383742108440E-005;
    t = t * i + 9.94570466342226000E-005;
    t = t * i - 2.23154589559238440E-004;
    t = t * i + 5.09669559149637430E-004;
    t = t * i - 1.19275392649162300E-003;
    t = t * i + 2.89051032936815490E-003;
    t = t * i - 7.38555102806811700E-003;
    t = t * i + 2.05808084278121250E-002;
    t = t * i - 6.73523010532073720E-002;
    t = t * i + 3.22467033424113040E-001;
    t = t * i + 4.22784335098467190E-001;
    t = t * i;
  } else if (fa >= 0.7) {
    i = 1.0 - fa;
    t =         1.17786911519331130E-002;  
    t = t * i + 3.89046747413522300E-002;
    t = t * i + 5.90045711362049900E-002;
    t = t * i + 6.02143305254344420E-002;
    t = t * i + 5.61652708964839180E-002;
    t = t * i + 5.75052755193461370E-002;
    t = t * i + 6.21061973447320710E-002;
    t = t * i + 6.67614724532521880E-002;
    t = t * i + 7.14856037245421020E-002;
    t = t * i + 7.69311251313347100E-002;
    t = t * i + 8.33503129714946310E-002;
    t = t * i + 9.09538288991182800E-002;
    t = t * i + 1.00099591546322310E-001;
    t = t * i + 1.11334278141734510E-001;
    t = t * i + 1.25509666613462880E-001;
    t = t * i + 1.44049896457704160E-001;
    t = t * i + 1.69557177031481600E-001;
    t = t * i + 2.07385551032182120E-001;
    t = t * i + 2.70580808427600350E-001;
    t = t * i + 4.00685634386517050E-001;
    t = t * i + 8.22467033424113540E-001;
    t = t * i + 5.77215664901532870E-001;
    t = t * i;
  } else {
    t =         -9.04051686831357990E-008;
    t = t * fa + 7.06814224969349250E-007;
    t = t * fa - 3.80702154637902830E-007;
    t = t * fa - 2.12880892189316100E-005;
    t = t * fa + 1.29108470307156190E-004;
    t = t * fa - 2.15932815215386580E-004;
    t = t * fa - 1.16484324388538480E-003;
    t = t * fa + 7.21883433044470670E-003;
    t = t * fa - 9.62194579514229560E-003;
    t = t * fa - 4.21977386992884450E-002;
    t = t * fa + 1.66538611813682460E-001;
    t = t * fa - 4.20026350606819980E-002;
    t = t * fa - 6.55878071519427450E-001;
    t = t * fa + 5.77215664901523870E-001;
    t = t * fa;
    t = t * fa + fa;
    t = -log (t);
  }
  if (a >= 0.0) return t;
  if (fa < 1e-19) return -log(fa);
  i = floor(fa);       
  if (fa == i) return 1.0 / (fa - i); /* a is an integer: return infinity */
  i = rint (2.0 * fa);
  quot = (long long int)i;
  i = fa - 0.5 * i;
  i = i * CUDART_PI;
  if (quot & 1) {
    i = cos(i);
  } else {
    i = sin(i);
  }
  i = fabs(i);
  t = log(CUDART_PI / (i * fa)) - t;
  return t;
}

__func__(unsigned long long int __internal_host_nan_kernel(const char *s))
{
  unsigned long long i = 0;
  int c;
  int ovfl = 0;
  int invld = 0;
  if (s && (*s == '0')) {
    s++;
    if ((*s == 'x') || (*s == 'X')) {
      s++; 
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x0fffffffffffffffULL) {
          ovfl = 1;
        }
        c = (((*s) >= 'A') && ((*s) <= 'F')) ? (*s + 'a' - 'A') : (*s);
        if ((c >= 'a') && (c <= 'f')) { 
          c = c - 'a' + 10;
          i = i * 16 + c;
        } else if ((c >= '0') && (c <= '9')) { 
          c = c - '0';
          i = i * 16 + c;
        } else {
          invld = 1;
        }
        s++;
      }
    } else {
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x1fffffffffffffffULL) {
          ovfl = 1;
        }
        c = *s;
        if ((c >= '0') && (c <= '7')) { 
          c = c - '0';
          i = i * 8 + c;
        } else {
          invld = 1; 
        }
        s++;
      }
    }
  } else if (s) {
    while (*s) {
      c = *s;
      if ((i > 1844674407370955161ULL) || 
          ((i == 1844674407370955161ULL) && (c > '5'))) {
        ovfl = 1;
      }
      if ((c >= '0') && (c <= '9')) { 
        c = c - '0';
        i = i * 10 + c;
      } else {
        invld = 1;
      }
      s++;
    }
  }
  if (ovfl) {
    i = ~0ULL;
  }
  if (invld) {
    i = 0ULL;
  }
  i = (i & 0x000fffffffffffffULL) | 0x7ff8000000000000ULL;
  return i;
}

__func__(double nan(const char *tagp))
{
  volatile union {
    unsigned long long l;
    double d;
  } cvt;

  cvt.l = __internal_host_nan_kernel(tagp);
  return cvt.d;
}

__func__(double __host_tgamma_kernel(double a))
{
  double t;
  t =       - 4.4268934071252475E-010;
  t = t * a - 2.0266591846658954E-007;
  t = t * a + 1.1381211721119527E-006;
  t = t * a - 1.2507734816630748E-006;
  t = t * a - 2.0136501740408771E-005;
  t = t * a + 1.2805012607354486E-004;
  t = t * a - 2.1524140811527418E-004;
  t = t * a - 1.1651675459704604E-003;
  t = t * a + 7.2189432248466381E-003;
  t = t * a - 9.6219715326862632E-003;
  t = t * a - 4.2197734554722394E-002;
  t = t * a + 1.6653861138250356E-001;
  t = t * a - 4.2002635034105444E-002;
  t = t * a - 6.5587807152025712E-001;
  t = t * a + 5.7721566490153287E-001;
  t = t * a + 1.0000000000000000E+000;
  return t;
}

__func__(double __host_stirling_poly(double a))
{
  double x = 1.0 / a;
  double z = 0.0;
  z =       + 8.3949872067208726e-004;
  z = z * x - 5.1717909082605919e-005;
  z = z * x - 5.9216643735369393e-004;
  z = z * x + 6.9728137583658571e-005;
  z = z * x + 7.8403922172006662e-004;
  z = z * x - 2.2947209362139917e-004;
  z = z * x - 2.6813271604938273e-003;
  z = z * x + 3.4722222222222220e-003;
  z = z * x + 8.3333333333333329e-002;
  z = z * x + 1.0000000000000000e+000;
  return z;
}

__func__(double __host_tgamma_stirling(double a))
{
  double z;
  double x;
  z = __host_stirling_poly (a);
  if (a < 142.0) {
    x = pow (a, a - 0.5);
    a = x * exp (-a);
    a = a * CUDART_SQRT_2PI;
    return a * z;
  } else if (a < 172.0) {
    x = pow (a, 0.5 * a - 0.25);
    a = x * exp (-a);
    a = a * CUDART_SQRT_2PI;
    a = a * z;
    return a * x;
  } else {
    return exp(1000.0); /* INF */
  }
}

__func__(double tgamma(double a))
{
  double s, xx, x = a;
  if (__isnan(a)) {
    return a + a;
  }
  if (fabs(x) < 20.0) {
    if (x >= 0.0) {
      s = 1.0;
      xx = x;
      while (xx > 1.5) {
        xx = xx - 1.0;
        s = s * xx;
      }
      if (x >= 0.5) {
        xx = xx - 1.0;
      }
      xx = __host_tgamma_kernel (xx);
      if (x < 0.5) {
        xx = xx * x;
      }
      s = s / xx;
    } else {
      xx = x;
      s = xx;
      if (x == floor(x)) {
        return 0.0 / (x - floor(x));
      }
      while (xx < -0.5) {
        xx = xx + 1.0;
        s = s * xx;
      }
      xx = __host_tgamma_kernel (xx);
      s = s * xx;
      s = 1.0 / s;
    }
    return s;
  } else {
    if (x >= 0.0) {
      return __host_tgamma_stirling (x);
    } else {
      double t;
      int quot;
      if (x == floor(x)) {
        return 0.0 / (x - floor(x));
      }
      if (x < -185.0) {
        int negative;
        x = floor(x);
        negative = ((x - (2.0 * floor(0.5 * x))) == 1.0);
        return negative ? (-1.0 / 1e308 / 1e308) : CUDART_ZERO;
      }
      /* compute sin(pi*x) accurately */
      xx = rint (2.0 * x);
      quot = (int)xx;
      xx = -0.5 * xx + x;
      xx = xx * CUDART_PI;
      if (quot & 1) {
        xx = cos (xx);
      } else {
        xx = sin (xx);
      }
      if (quot & 2) {
        xx = -xx;
      }
      x = fabs (x);
      s = exp (-x);
      t = x - 0.5;
      if (x > 140.0) t = 0.5 * t;
      t = pow (x, t);
      if (x > 140.0) s = s * t;
      s = s * __host_stirling_poly (x);
      s = s * x;
      s = s * xx;
      s = 1.0 / s;
      s = s * CUDART_SQRT_PIO2;
      s = s / t;
      return s;
    }
  }
}
#endif /* _MSC_VER < 1800 */

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR FLOAT AND LONG DOUBLE ROUTINES FOR WINDOWS PLATFORM  *
* MAP FLOAT AND LONG DOUBLE ROUTINES TO DOUBLE ROUTINES                        *
*                                                                              *
*******************************************************************************/

__func__(int __signbitl(long double a))
{
  return __signbit((double)a);
}

__func__(int __signbitf(float a))
{
  return __signbit((double)a);
}

__func__(int __finitel(long double a))
{
  return __finite((double)a);
}

__func__(int __finitef(float a))
{
  return __finite((double)a);
}

__func__(int __isinfl(long double a))
{
  return __isinf((double)a);
}

__func__(int __isinff(float a))
{
  return __isinf((double)a);
}

__func__(int __isnanl(long double a))
{
  return __isnan((double)a);
}

__func__(int __isnanf(float a))
{
  return __isnan((double)a);
}

#if _MSC_VER < 1800
__func__(float fmaxf(float a, float b))
{
  return (float)fmax((double)a, (double)b);
}

__func__(float fminf(float a, float b))
{
  return (float)fmin((double)a, (double)b);
}

__func__(float roundf(float a))
{
  return (float)round((double)a);
}

__func__(long int lroundf(float a))
{
  return lround((double)a);
}

__func__(long long int llroundf(float a))
{
  return llround((double)a);
}

__func__(float truncf(float a))
{
  return (float)trunc((double)a);
}

__func__(float rintf(float a))
{
  return (float)rint((double)a);
}

__func__(float nearbyintf(float a))
{
  return (float)nearbyint((double)a);
}

__func__(long int lrintf(float a))
{
  return lrint((double)a);
}

__func__(long long int llrintf(float a))
{
  return llrint((double)a);
}

__func__(float logbf(float a))
{
  return (float)logb((double)a);
}

__func__(float scalblnf(float a, long int b))
{
  return (float)scalbln((double)a, b);
}

__func__(float log2f(float a))
{
  return (float)log2((double)a);
}

__func__(float exp2f(float a))
{
  return (float)exp2((double)a);
}

__func__(float acoshf(float a))
{
  return (float)acosh((double)a);
}

__func__(float asinhf(float a))
{
  return (float)asinh((double)a);
}

__func__(float atanhf(float a))
{
  return (float)atanh((double)a);
}

__func__(float cbrtf(float a))
{
  return (float)cbrt((double)a);
}

__func__(float expm1f(float a))
{
  return (float)expm1((double)a);
}

__func__(float fdimf(float a, float b))
{
  return (float)fdim((double)a, (double)b);
}

__func__(float log1pf(float a))
{
  return (float)log1p((double)a);
}

__func__(float scalbnf(float a, int b))
{
  return (float)scalbn((double)a, b);
}

__func__(float fmaf(float a, float b, float c))
{
  return (float)fma((double)a, (double)b, (double)c);
}

__func__(int ilogbf(float a))
{
  return ilogb((double)a);
}

__func__(float erff(float a))
{
  return (float)erf((double)a);
}

__func__(float erfcf(float a))
{
  return (float)erfc((double)a);
}

__func__(float lgammaf(float a))
{
  return (float)lgamma((double)a);
}

__func__(float tgammaf(float a))
{
  return (float)tgamma((double)a);
}

__func__(float remquof(float a, float b, int *quo))
{
  return (float)remquo((double)a, (double)b, quo);
}

__func__(float remainderf(float a, float b))
{
  return (float)remainder((double)a, (double)b);
}
#endif /* _MSC_VER < 1800 */

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR FLOAT ROUTINES FOR WINDOWS PLATFORM                  *
*                                                                              *
*******************************************************************************/

#if _MSC_VER < 1800
__func__(float copysignf(float a, float b))
{
  volatile union {
    float f;
    unsigned int i;
  } aa, bb;

  aa.f = a;
  bb.f = b;
  aa.i = (aa.i & ~0x80000000) | (bb.i & 0x80000000);
  return aa.f;
}

__func__(float nextafterf(float a, float b))
{
  volatile union {
    float f;
    unsigned int i;
  } cvt;
  unsigned int ia;
  unsigned int ib;
  cvt.f = a;
  ia = cvt.i;
  cvt.f = b;
  ib = cvt.i;
  if (__isnanf(a) || __isnanf(b)) return a + b; /*NaN*/
  if (((ia | ib) << 1) == 0) return b;
  if (a == 0.0f) {
    return copysignf(1.401298464e-045f, b); /*crossover*/
  }
  if ((a < b) && (a < 0.0f)) ia--;
  if ((a < b) && (a > 0.0f)) ia++;
  if ((a > b) && (a < 0.0f)) ia++;
  if ((a > b) && (a > 0.0f)) ia--;
  cvt.i = ia;
  return cvt.f;
}

__func__(float nanf(const char *tagp))
{
  volatile union {
    float f;
    unsigned int i;
  } cvt;
  
  cvt.i = (unsigned int)__internal_host_nan_kernel(tagp);
  cvt.i = (cvt.i & 0x007fffff) | 0x7fc00000;
  return cvt.f;
}

#endif /* _MSC_VER < 1800 */

#endif /* _WIN32 */

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR DOUBLE AND FLOAT ROUTINES. ALL PLATFORMS             *
*                                                                              *
*******************************************************************************/

__func__(double rsqrt(double a))
{
  return 1.0 / sqrt(a);
}

__func__(double rcbrt(double a))
{
  double s, t;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return 1.0 / a;
  } 
  s = fabs(a);
  t = exp2(-CUDART_THIRD * log2(s));                /* initial approximation */
  t = ((t*t) * (-s*t) + 1.0) * (CUDART_THIRD*t) + t;/* refine approximation */
#if defined(__APPLE__)
  if (__signbitd(a))
#else /* __APPLE__ */
  if (__signbit(a))
#endif /* __APPLE__ */
  {
    t = -t;
  }
  return t;
}

__func__(double sinpi(double a))
{
  int n;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return sin (a);
  } 
  if (a == floor(a)) {
    return ((a / 1.0e308) / 1.0e308) / 1.0e308;
  }
  a = remquo (a, 0.5, &n);
  a = a * CUDART_PI;
  if (n & 1) {
    a = cos (a);
  } else {
    a = sin (a);
  }
  if (n & 2) {
    a = -a;
  }
  return a;
}

__func__(double cospi(double a))
{
  int n;

  if (__isnan(a)) {
    return a + a;
  }
  if (__isinf(a)) {
    return cos (a);
  } 
  if (fabs(a) > 9.0071992547409920e+015) {
    a = 0.0;
  }
  a = remquo (a, 0.5, &n);
  a = a * CUDART_PI;
  n++;
  if (n & 1) {
    a = cos (a);
  } else {
    a = sin (a);
  }
  if (n & 2) {
    a = -a;
  }
  if (a == 0.0) {
    a = fabs(a);
  }
  return a;
}

__func__(void sincospi(double a, double *sptr, double *cptr))
{
  *sptr = sinpi(a);
  *cptr = cospi(a);
}

__func__(double erfinv(double a))
{
  double p, q, t, fa;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  fa = fabs(a);
  if (fa >= 1.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;                    /* INDEFINITE */
    if (fa == 1.0) {
      t = a * exp(1000.0);        /* Infinity */
    }
  } else if (fa >= 0.9375) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
     */
    t = log1p(-fa);
    t = 1.0 / sqrt(-t);
    p =         2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q =     t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 39
    */
    t = a * a - .87890625;
    p =         .21489185007307062000e+0;
    p = p * t - .64200071507209448655e+1;
    p = p * t + .29631331505876308123e+2;
    p = p * t - .47644367129787181803e+2;
    p = p * t + .34810057749357500873e+2;
    p = p * t - .12954198980646771502e+2;
    p = p * t + .25349389220714893917e+1;
    p = p * t - .24758242362823355486e+0;
    p = p * t + .94897362808681080020e-2;
    q =     t - .12831383833953226499e+2;
    q = q * t + .41409991778428888716e+2;
    q = q * t - .53715373448862143349e+2;
    q = q * t + .33880176779595142685e+2;
    q = q * t - .11315360624238054876e+2;
    q = q * t + .20369295047216351160e+1;
    q = q * t - .18611650627372178511e+0;
    q = q * t + .67544512778850945940e-2;
    p = p / q;
    t = a * p;
  } else {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 18
    */
    t = a * a - .5625;
    p =       - .23886240104308755900e+2;
    p = p * t + .45560204272689128170e+3;
    p = p * t - .22977467176607144887e+4;
    p = p * t + .46631433533434331287e+4;
    p = p * t - .43799652308386926161e+4;
    p = p * t + .19007153590528134753e+4;
    p = p * t - .30786872642313695280e+3;
    q =     t - .83288327901936570000e+2;
    q = q * t + .92741319160935318800e+3;
    q = q * t - .35088976383877264098e+4;
    q = q * t + .59039348134843665626e+4;
    q = q * t - .48481635430048872102e+4;
    q = q * t + .18997769186453057810e+4;
    q = q * t - .28386514725366621129e+3;
    p = p / q;
    t = a * p;
  }
  return t;
}

__func__(double erfcinv(double a))
{
  double t;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  if (__isnan(a)) {
    return a + a;
  }
  if (a <= 0.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;                        /* INDEFINITE */
    if (a == 0.0) {
        t = (1.0 - a) * exp(1000.0);  /* Infinity */
    }
  } 
  else if (a >= 0.0625) {
    t = erfinv (1.0 - a);
  }
  else if (a >= 1e-100) {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
    */
    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p =         2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q =     t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
  }
  else {
    /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
       Approximations for the Inverse of the Error Function. Mathematics of
       Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 82
    */
    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p =         6.9952990607058154858e-1;
    p = p * t + 1.9507620287580568829e+0;
    p = p * t + 8.2810030904462690216e-1;
    p = p * t + 1.1279046353630280005e-1;
    p = p * t + 6.0537914739162189689e-3;
    p = p * t + 1.3714329569665128933e-4;
    p = p * t + 1.2964481560643197452e-6;
    p = p * t + 4.6156006321345332510e-9;
    p = p * t + 4.5344689563209398450e-12;
    q =     t + 1.5771922386662040546e+0;
    q = q * t + 2.1238242087454993542e+0;
    q = q * t + 8.4001814918178042919e-1;
    q = q * t + 1.1311889334355782065e-1;
    q = q * t + 6.0574830550097140404e-3;
    q = q * t + 1.3715891988350205065e-4;
    q = q * t + 1.2964671850944981713e-6;
    q = q * t + 4.6156017600933592558e-9;
    q = q * t + 4.5344687377088206783e-12;
    t = p / (q * t);
  }
  return t;
}

__func__(double normcdfinv(double a))
{
  return -1.4142135623730951 * erfcinv(a + a);
}

__func__(double normcdf(double a))
{
  double ah, al, t1, t2, u1, u2, v1, v2, z;
  if (fabs (a) > 38.5) a = copysign (38.5, a);
  ah = a * 134217729.0;
  u1 = (a - ah) + ah;
  u2 = a - u1;
  v1 = -7.0710678398609161e-01;
  v2 =  2.7995440410322203e-09;
  t1 = a * -CUDART_SQRT_HALF_HI;
  t2 = (((u1 * v1 - t1) + u1 * v2) + u2 * v1) + u2 * v2;
  t2 = (a * -CUDART_SQRT_HALF_LO) + t2;
  ah = t1 + t2;
  z = erfc (ah);
  if (a < -1.0) {
    al = (t1 - ah) + t2;
    t1 = -2.0 * ah * z;
    z = t1 * al + z;
  }
  return 0.5 * z;
}

__func__(double erfcx(double a))
{
  double x, t1, t2, t3;

  if (__isnan(a)) {
    return a + a;
  }
  x = fabs(a); 
  if (x < 32.0) {
    /*  
     * This implementation of erfcx() is based on the algorithm in: M. M. 
     * Shepherd and J. G. Laframboise, "Chebyshev Approximation of (1 + 2x)
     * exp(x^2)erfc x in 0 <= x < INF", Mathematics of Computation, Vol. 
     * 36, No. 153, January 1981, pp. 249-253. For the core approximation,
     * the input domain [0,INF] is transformed via (x-k) / (x+k) where k is
     * a precision-dependent constant. Here, we choose k = 4.0, so the input 
     * domain [0, 27.3] is transformed into the core approximation domain 
     * [-1, 0.744409].   
     */  
    /* (1+2*x)*exp(x*x)*erfc(x) */ 
    /* t2 = (x-4.0)/(x+4.0), transforming [0,INF] to [-1,+1] */ 
    t1 = x - 4.0; 
    t2 = x + 4.0; 
    t2 = t1 / t2;
    /* approximate on [-1, 0.744409] */   
    t1 =         - 3.5602694826817400E-010; 
    t1 = t1 * t2 - 9.7239122591447274E-009; 
    t1 = t1 * t2 - 8.9350224851649119E-009; 
    t1 = t1 * t2 + 1.0404430921625484E-007; 
    t1 = t1 * t2 + 5.8806698585341259E-008; 
    t1 = t1 * t2 - 8.2147414929116908E-007; 
    t1 = t1 * t2 + 3.0956409853306241E-007; 
    t1 = t1 * t2 + 5.7087871844325649E-006; 
    t1 = t1 * t2 - 1.1231787437600085E-005; 
    t1 = t1 * t2 - 2.4399558857200190E-005; 
    t1 = t1 * t2 + 1.5062557169571788E-004; 
    t1 = t1 * t2 - 1.9925637684786154E-004; 
    t1 = t1 * t2 - 7.5777429182785833E-004; 
    t1 = t1 * t2 + 5.0319698792599572E-003; 
    t1 = t1 * t2 - 1.6197733895953217E-002; 
    t1 = t1 * t2 + 3.7167515553018733E-002; 
    t1 = t1 * t2 - 6.6330365827532434E-002; 
    t1 = t1 * t2 + 9.3732834997115544E-002; 
    t1 = t1 * t2 - 1.0103906603555676E-001; 
    t1 = t1 * t2 + 6.8097054254735140E-002; 
    t1 = t1 * t2 + 1.5379652102605428E-002; 
    t1 = t1 * t2 - 1.3962111684056291E-001; 
    t1 = t1 * t2 + 1.2329951186255526E+000; 
    /* (1+2*x)*exp(x*x)*erfc(x) / (1+2*x) = exp(x*x)*erfc(x) */  
    t2 = 2.0 * x + 1.0; 
    t1 = t1 / t2;
  } else {
    /* asymptotic expansion for large aguments */
    t2 = 1.0 / x;
    t3 = t2 * t2;
    t1 =         -29.53125;
    t1 = t1 * t3 + 6.5625;
    t1 = t1 * t3 - 1.875;
    t1 = t1 * t3 + 0.75;
    t1 = t1 * t3 - 0.5;
    t1 = t1 * t3 + 1.0;
    t2 = t2 * 5.6418958354775628e-001;
    t1 = t1 * t2;
  }
  if (a < 0.0) {
    /* erfcx(x) = 2*exp(x^2) - erfcx(|x|) */
    t2 = ((int)(x * 16.0)) * 0.0625;
    t3 = (x - t2) * (x + t2);
    t3 = exp(t2 * t2) * exp(t3);
    t3 = t3 + t3;
    t1 = t3 - t1;
  }
  return t1;
}

__func__(float rsqrtf(float a))
{
  return (float)rsqrt((double)a);
}

__func__(float rcbrtf(float a))
{
  return (float)rcbrt((double)a);
}

__func__(float sinpif(float a))
{
  return (float)sinpi((double)a);
}

__func__(float cospif(float a))
{
  return (float)cospi((double)a);
}

__func__(void sincospif(float a, float *sptr, float *cptr))
{
  double s, c;

  sincospi((double)a, &s, &c);
  *sptr = (float)s;
  *cptr = (float)c;
}

__func__(float erfinvf(float a))
{
  return (float)erfinv((double)a);
}

__func__(float erfcinvf(float a))
{
  return (float)erfcinv((double)a);
}

__func__(float normcdfinvf(float a))
{
  return (float)normcdfinv((double)a);
}

__func__(float normcdff(float a))
{
  return (float)normcdf((double)a);
}

__func__(float erfcxf(float a))
{
  return (float)erfcx((double)a);
}

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATION FOR UTILITY ROUTINES. ALL PLATFORMS                      *
*                                                                              *
*******************************************************************************/

__func__(int min(int a, int b))
{
  return a < b ? a : b;
}

__func__(unsigned int umin(unsigned int a, unsigned int b))
{
  return a < b ? a : b;
}

__func__(long long int llmin(long long int a, long long int b))
{
  return a < b ? a : b;
}

__func__(unsigned long long int ullmin(unsigned long long int a, unsigned long long int b))
{
  return a < b ? a : b;
}

__func__(int max(int a, int b))
{
  return a > b ? a : b;
}

__func__(unsigned int umax(unsigned int a, unsigned int b))
{
  return a > b ? a : b;
}

__func__(long long int llmax(long long int a, long long int b))
{
  return a > b ? a : b;
}

__func__(unsigned long long int ullmax(unsigned long long int a, unsigned long long int b))
{
  return a > b ? a : b;
}

#if defined(_WIN32)

#pragma warning(default: 4211)

#endif /* _WIN32 */

#endif /* __CUDACC_RTC__ || __CUDABE__ */

#endif /* __CUDACC_RTC__ || !__CUDACC__ */

#endif /* !__MATH_FUNCTIONS_HPP__ */

