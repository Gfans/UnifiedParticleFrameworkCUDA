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

#if !defined(__MATH_FUNCTIONS_DBL_PTX3_H__)
#define __MATH_FUNCTIONS_DBL_PTX3_H__

#if defined(__CUDACC_RTC__)
#define __MATH_FUNCTIONS_DBL_PTX3_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DBL_PTX3_DECL__ static __forceinline__
#endif /* __CUDACC_RTC__ */

/* True double precision implementations, since native double support */

#if defined(__CUDABE__) || defined(__CUDACC_RTC__)

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rint(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long int lrint(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long long int llrint(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nearbyint(double a);

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitd(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isfinited(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinfd(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnand(double a);

#if defined(__APPLE__)

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isfinite(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinf(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnan(/* we do not support long double yet, hence double */double a);

#else /* __APPLE__ */

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbit(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __finite(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __finitel(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinf(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinfl(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnan(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnanl(/* we do not support long double yet, hence double */double a);

#endif /* __APPLE__ */

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double copysign(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ void sincos(double a, double *sptr, double *cptr);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ void sincospi(double a, double *sptr, double *cptr);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sin(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cos(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sinpi(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cospi(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tan(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log2(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log10(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log1p(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp2(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp10(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double expm1(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cosh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sinh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tanh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atan2(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atan(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double asin(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double acos(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double acosh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double asinh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atanh(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double hypot(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rhypot(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double norm3d(double a, double b, double c);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rnorm3d(double a, double b, double c);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double norm4d(double a, double b, double c, double d);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cbrt(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rcbrt(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double pow(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double j0(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double j1(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double y0(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double y1(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double yn(int n, double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double jn(int n, double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cyl_bessel_i0(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cyl_bessel_i1(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erf(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfinv(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfcinv(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double normcdfinv(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfc(double a)  ;

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfcx(double a)  ;

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double normcdf(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tgamma(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double lgamma(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double ldexp(double a, int b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double scalbn(double a, int b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double scalbln(double a, long int b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double frexp(double a, int *b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double modf(double a, double *b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fmod(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double remainder(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double remquo(double a, double b, int *c);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nextafter(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nan(const char *tagp);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double round(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long long int llround(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long int lround(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fdim(double a, double b);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int ilogb(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double logb(double a);

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fma(double a, double b, double c);

#endif /* __CUDABE__ || __CUDACC_RTC__ */

#undef __MATH_FUNCTIONS_DBL_PTX3_DECL__

#if !defined(__CUDACC_RTC__)
#include "math_functions_dbl_ptx3.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* __MATH_FUNCTIONS_DBL_PTX3_H__ */
