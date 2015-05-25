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

#if !defined(__MATH_FUNCTIONS_DBL_PTX3_HPP__)
#define __MATH_FUNCTIONS_DBL_PTX3_HPP__

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

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rint(double a)
{
  return __nv_rint(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long int lrint(double a)
{
#if defined(__LP64__)
  return (long int)__double2ll_rn(a);
#else /* __LP64__ */
  return (long int)__double2int_rn(a);
#endif /* __LP64__ */
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long long int llrint(double a)
{
  return __nv_llrint(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nearbyint(double a)
{
  return __nv_nearbyint(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitd(double a)
{
  return __nv_signbitd(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isfinited(double a)
{
  return __nv_isfinited(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinfd(double a)
{
  return __nv_isinfd(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnand(double a)
{
  return __nv_isnand(a);
}

#if defined(__APPLE__)

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a)
{
  return __signbitd((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isfinite(/* we do not support long double yet, hence double */double a)
{
  return __isfinited((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinf(/* we do not support long double yet, hence double */double a)
{
  return __isinfd((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnan(/* we do not support long double yet, hence double */double a)
{
  return __isnand((double)a);
}

#else /* __APPLE__ */

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbit(double a)
{
  return __signbitd(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a)
{
  return __signbit((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __finite(double a)
{
  return __isfinited(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __finitel(/* we do not support long double yet, hence double */double a)
{
  return __finite((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinf(double a)
{
  return __isinfd(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isinfl(/* we do not support long double yet, hence double */double a)
{
  return __isinf((double)a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnan(double a)
{
  return __isnand(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int __isnanl(/* we do not support long double yet, hence double */double a)
{
  return __isnan((double)a);
}

#endif /* __APPLE__ */

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double copysign(double a, double b)
{
  return __nv_copysign(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ void sincos(double a, double *sptr, double *cptr)
{
  __nv_sincos(a, sptr, cptr);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ void sincospi(double a, double *sptr, double *cptr)
{
  __nv_sincospi(a, sptr, cptr);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sin(double a)
{
  return __nv_sin(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cos(double a)
{
  return __nv_cos(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sinpi(double a)
{
  return __nv_sinpi(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cospi(double a)
{
  return __nv_cospi(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tan(double a)
{
  return __nv_tan(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log(double a)
{
  return __nv_log(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log2(double a)
{
  return __nv_log2(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log10(double a)
{
  return __nv_log10(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double log1p(double a)
{
  return __nv_log1p(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp(double a)
{
  return __nv_exp(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp2(double a)
{
  return __nv_exp2(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double exp10(double a)
{
  return __nv_exp10(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double expm1(double a)
{
  return __nv_expm1(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cosh(double a)
{
  return __nv_cosh(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double sinh(double a)
{
  return __nv_sinh(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tanh(double a)
{
  return __nv_tanh(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atan2(double a, double b)
{
  return __nv_atan2(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atan(double a)
{
  return __nv_atan(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double asin(double a)
{
  return __nv_asin(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double acos(double a)
{
  return __nv_acos(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double acosh(double a)
{
  return __nv_acosh(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double asinh(double a)
{
  return __nv_asinh(a);  
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double atanh(double a)
{
  return __nv_atanh(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double hypot(double a, double b)
{
  return __nv_hypot(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rhypot(double a, double b)
{
  return __nv_rhypot(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double norm3d(double a, double b, double c)
{
  return __nv_norm3d(a, b, c);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rnorm3d(double a, double b, double c)
{
  return __nv_rnorm3d(a, b, c);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double norm4d(double a, double b, double c, double d)
{
  return __nv_norm4d(a, b, c, d);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cbrt(double a)
{
  return __nv_cbrt(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double rcbrt(double a)
{
  return __nv_rcbrt(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double pow(double a, double b)
{
  return __nv_pow(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double j0(double a)
{
  return __nv_j0(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double j1(double a)
{
  return __nv_j1(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double y0(double a)
{
  return __nv_y0(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double y1(double a)
{
  return __nv_y1(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double yn(int n, double a)
{
  return __nv_yn(n, a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double jn(int n, double a)
{
  return __nv_jn(n, a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cyl_bessel_i0(double a)
{
  return __nv_cyl_bessel_i0(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double cyl_bessel_i1(double a)
{
  return __nv_cyl_bessel_i1(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erf(double a)
{
  return __nv_erf(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfinv(double a)
{
  return __nv_erfinv(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfcinv(double a)
{
  return __nv_erfcinv(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double normcdfinv(double a)
{
  return __nv_normcdfinv(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfc(double a)  
{  
  return __nv_erfc(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double erfcx(double a)  
{
  return __nv_erfcx(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double normcdf(double a)
{
  return __nv_normcdf(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double tgamma(double a)
{
  return __nv_tgamma(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double lgamma(double a)
{
  return __nv_lgamma(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double ldexp(double a, int b)
{
  return __nv_ldexp(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double scalbn(double a, int b)
{
  return __nv_scalbn(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double scalbln(double a, long int b)
{
#if defined(__LP64__)
  /* clamp to integer range prior to conversion */
  if (b < -2147483648L) b = -2147483648L;
  if (b >  2147483647L) b =  2147483647L;
#endif /* __LP64__ */
  return scalbn(a, (int)b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double frexp(double a, int *b)
{
  return __nv_frexp(a, b);  
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double modf(double a, double *b)
{
  return __nv_modf(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fmod(double a, double b)
{
  return __nv_fmod(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double remainder(double a, double b)
{
  return __nv_remainder(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double remquo(double a, double b, int *c)
{
  return __nv_remquo(a, b, c);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nextafter(double a, double b)
{
  return __nv_nextafter(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double nan(const char *tagp)
{
  return __nv_nan((const signed char *) tagp);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double round(double a)
{
  return __nv_round(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long long int llround(double a)
{
  return __nv_llround(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ long int lround(double a)
{
#if defined(__LP64__)
  return (long int)llround(a);
#else /* __LP64__ */
  return (long int)round(a);
#endif /* __LP64__ */
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fdim(double a, double b)
{
  return __nv_fdim(a, b);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ int ilogb(double a)
{
  return __nv_ilogb(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double logb(double a)
{
  return __nv_logb(a);
}

__MATH_FUNCTIONS_DBL_PTX3_DECL__ double fma(double a, double b, double c)
{
  return __nv_fma(a, b, c);
}

#endif /* __CUDABE__ || __CUDACC_RTC__ */

#undef __MATH_FUNCTIONS_DBL_PTX3_DECL__

#endif /* __MATH_FUNCTIONS_DBL_PTX3_HPP__ */

