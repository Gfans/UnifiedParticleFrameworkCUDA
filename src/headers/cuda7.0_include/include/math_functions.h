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

#if !defined(__MATH_FUNCTIONS_H__)
#define __MATH_FUNCTIONS_H__

/**
 * \defgroup CUDA_MATH Mathematical Functions
 *
 * CUDA mathematical functions are always available in device code.
 * Some functions are also available in host code as indicated.
 *
 * Note that floating-point functions are overloaded for different
 * argument types.  For example, the ::log() function has the following
 * prototypes:
 * \code
 * double log(double x);
 * float log(float x);
 * float logf(float x);
 * \endcode
 */

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

#if !defined(__CUDACC_RTC__)
extern "C"
{
#endif /* !__CUDACC_RTC__ */

/* Define math function DOXYGEN toplevel groups, functions will
   be added to these groups later.
*/
/**
 * \defgroup CUDA_MATH_SINGLE Single Precision Mathematical Functions
 * This section describes single precision mathematical functions.
 */

/**
 * \defgroup CUDA_MATH_DOUBLE Double Precision Mathematical Functions
 * This section describes double precision mathematical functions.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_SINGLE Single Precision Intrinsics
 * This section describes single precision intrinsic functions that are
 * only supported in device code.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_DOUBLE Double Precision Intrinsics
 * This section describes double precision intrinsic functions that are
 * only supported in device code.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_INT Integer Intrinsics
 * This section describes integer intrinsic functions that are
 * only supported in device code.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_CAST Type Casting Intrinsics
 * This section describes type casting intrinsic functions that are
 * only supported in device code.
 */

/**
 *
 * \defgroup CUDA_MATH_INTRINSIC_SIMD SIMD Intrinsics
 * This section describes SIMD intrinsic functions that are
 * only supported in device code.
 */


/**
 * @}
 */

#if defined(__ANDROID__) && (__ANDROID_API__ <= 20) && !defined(__aarch64__)
static __host__ __device__ __device_builtin__ __cudart_builtin__ int                    abs(int);
static __host__ __device__ __device_builtin__ __cudart_builtin__ long int               labs(long int);
static __host__ __device__ __device_builtin__ __cudart_builtin__ long long int          llabs(long long int);
#else /* __ANDROID__ */
extern __host__ __device__ __device_builtin__ __cudart_builtin__ int            __cdecl abs(int) __THROW;
extern __host__ __device__ __device_builtin__ __cudart_builtin__ long int       __cdecl labs(long int) __THROW;
extern __host__ __device__ __device_builtin__ __cudart_builtin__ long long int          llabs(long long int) __THROW;
#endif /* __ANDROID__ */

#ifdef __QNX__
/* put all math functions in std */
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the absolute value of the input argument.
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of the input argument.
 * - fabs(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fabs(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl fabs(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the absolute value of its argument
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of its argument.
 * - fabs(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fabs(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  fabsf(float x) __THROW;
#ifdef __QNX__
} /* std */
#endif
extern __host__ __device__ __device_builtin__ int                    min(int, int);

extern __host__ __device__ __device_builtin__ unsigned int           umin(unsigned int, unsigned int);
extern __host__ __device__ __device_builtin__ long long int          llmin(long long int, long long int);
extern __host__ __device__ __device_builtin__ unsigned long long int ullmin(unsigned long long int, unsigned long long int);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  fminf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fminf(float x, float y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 fmin(double x, double y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl fmin(double x, double y);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif
extern __host__ __device__ __device_builtin__ int                    max(int, int);

extern __host__ __device__ __device_builtin__ unsigned int           umax(unsigned int, unsigned int);
extern __host__ __device__ __device_builtin__ long long int          llmax(long long int, long long int);
extern __host__ __device__ __device_builtin__ unsigned long long int ullmax(unsigned long long int, unsigned long long int);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  fmaxf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fmaxf(float x, float y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 fmax(double, double) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl fmax(double, double);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return 
 * - sin(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sin(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl sin(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return 
 * - cos(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cos(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl cos(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured 
 * in radians). The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sin() and ::cos().
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ void                   sincos(double x, double *sptr, double *cptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the second 
 * argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinf() and ::cosf().
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ void                   sincosf(float x, float *sptr, float *cptr) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return 
 * - tan(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tan(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl tan(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x, 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl sqrt(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x, 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - rsqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 rsqrt(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x, 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - rsqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  rsqrtf(float x);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return 
 * - log2(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log2(1) returns +0.
 * - log2(\p x) returns NaN for \p x < 0.
 * - log2(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 log2(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl log2(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 exp2(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl exp2(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  exp2f(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl exp2f(float x);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */         
extern __host__ __device__ __device_builtin__ double                 exp10(double x) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ float                  exp10f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return Returns 
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 expm1(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl expm1(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return  Returns 
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  expm1f(float x) __THROW;        
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl expm1f(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return
 * - log2f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log2f(1) returns +0.
 * - log2f(\p x) returns NaN for \p x < 0.
 * - log2f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  log2f(float x) __THROW;         
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl log2f(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return 
 * - log10(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log10(1) returns +0.
 * - log10(\p x) returns NaN for \p x < 0.
 * - log10(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl log10(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument \p x.
 *
 * \return 
 * - log(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log(1) returns +0.
 * - log(\p x) returns NaN for \p x < 0.
 * - log(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly

 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl log(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return 
 * - log1p(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log1p(-1) returns +0.
 * - log1p(\p x) returns NaN for \p x < -1.
 * - log1p(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 log1p(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl log1p(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return 
 * - log1pf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log1pf(-1) returns +0.
 * - log1pf(\p x) returns NaN for \p x < -1.
 * - log1pf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  log1pf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl log1pf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the largest integer less than or equal to \p x.
 * 
 * Calculates the largest integer value which is less than or equal to \p x.
 * 
 * \return
 * Returns 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - floor(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - floor(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl floor(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl exp(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * - cosh(0) returns 1.
 * - cosh(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl cosh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - sinh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl sinh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - tanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl tanh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * Result will be in the interval [0, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - acosh(1) returns 0.
 * - acosh(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 acosh(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl acosh(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * Result will be in the interval [0, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - acoshf(1) returns 0.
 * - acoshf(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  acoshf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl acoshf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - asinh(0) returns 1.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 asinh(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl asinh(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - asinhf(0) returns 1.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  asinhf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl asinhf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - atanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanh(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanh(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 atanh(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl atanh(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - atanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanhf(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanhf(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  atanhf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl atanhf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return 
 * - ldexp(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl ldexp(double x, int exp) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return 
 * - ldexpf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating point range.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  ldexpf(float x, int exp) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the floating point representation of the exponent of the input argument.
 *
 * Calculate the floating point representation of the exponent of the input argument \p x.
 *
 * \return 
 * - logb
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - logb
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 logb(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl logb(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating point representation of the exponent of the input argument.
 *
 * Calculate the floating point representation of the exponent of the input argument \p x.
 *
 * \return 
 * - logbf
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - logbf
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  logbf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl logbf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - ilogb(0) returns <tt>INT_MIN</tt>.
 * - ilogb(NaN) returns NaN.
 * - ilogb(\p x) returns <tt>INT_MAX</tt> if \p x is 
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - ilogb(\p x) return <tt>INT_MIN</tt> if the correct value is less 
 *     than <tt>INT_MIN</tt>.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ int                    ilogb(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP int    __cdecl ilogb(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - ilogbf(0) returns <tt>INT_MIN</tt>.
 * - ilogbf(NaN) returns NaN.
 * - ilogbf(\p x) returns <tt>INT_MAX</tt> if \p x is 
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - ilogbf(\p x) return <tt>INT_MIN</tt> if the correct value is less 
 *     than <tt>INT_MIN</tt>.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ int                    ilogbf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP int    __cdecl ilogbf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbn(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbn(\p x, 0) returns \p x.
 * - scalbn(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 scalbn(double x, int n) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl scalbn(double x, int n);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbnf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbnf(\p x, 0) returns \p x.
 * - scalbnf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  scalbnf(float x, int n) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl scalbnf(float x, int n);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbln(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbln(\p x, 0) returns \p x.
 * - scalbln(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 scalbln(double x, long int n) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl scalbln(double x, long int n);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalblnf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalblnf(\p x, 0) returns \p x.
 * - scalblnf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  scalblnf(float x, long int n) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl scalblnf(float x, long int n);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Extract mantissa and exponent of a floating-point value
 * 
 * Decompose the floating-point value \p x into a component \p m for the 
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and 
 * less than 1.0 or it will be equal to 0; 
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - frexp(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - frexp(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - frexp(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the 
 * location to which \p nptr points.
 * - frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl frexp(double x, int *nptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Extract mantissa and exponent of a floating-point value
 * 
 * Decomposes the floating-point value \p x into a component \p m for the 
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and 
 * less than 1.0 or it will be equal to 0; 
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - frexp(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - frexp(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - frexp(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the 
 * location to which \p nptr points.
 * - frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  frexpf(float x, int *nptr) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rint().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 round(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl round(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rintf().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  roundf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl roundf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::lrint().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long int               lround(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long int __cdecl lround(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::lrintf().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long int               lroundf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long int __cdecl lroundf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrint().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long long int          llround(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long long int __cdecl llround(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrintf().
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long long int          llroundf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long long int __cdecl llroundf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 rint(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl rint(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded towards zero.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  rintf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl rintf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long int               lrint(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long int __cdecl lrint(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long int               lrintf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long int __cdecl lrintf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long long int          llrint(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long long int __cdecl llrint(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ long long int          llrintf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP long long int __cdecl llrintf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in double precision floating-point format.
 *
 * \return 
 * - nearbyint(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - nearbyint(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 nearbyint(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl nearbyint(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in single precision floating-point format.
 *
 * \return 
 * - nearbyintf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - nearbyintf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  nearbyintf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl nearbyintf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns 
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- ⌈ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- ⌉ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 expressed as a floating-point number.
 * - ceil(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ceil(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl ceil(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in 
 * magnitude.
 *
 * \return 
 * Returns truncated integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 trunc(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl trunc(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in 
 * magnitude.
 *
 * \return 
 * Returns truncated integer value.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  truncf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl truncf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return 
 * Returns the positive difference between \p x and \p y.
 * - fdim(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - fdim(\p x, \p y) returns +0 if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 fdim(double x, double y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl fdim(double x, double y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return 
 * Returns the positive difference between \p x and \p y.
 * - fdimf(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - fdimf(\p x, \p y) returns +0 if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  fdimf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fdimf(float x, float y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p y / \p x. The quadrant of the result is
 * determined by the signs of inputs \p y and \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atan2(0, 1) returns +0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl atan2(double y, double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2].
 * - atan(0) returns +0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl atan(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [0, 
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - acos(1) returns +0.
 * - acos(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl acos(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2] for \p x inside [-1, +1].
 * - asin(0) returns +0.
 * - asin(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl asin(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculate the length of the hypotenuse of a right triangle whose two sides have lengths 
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse 
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
#if defined(_WIN32)
static __host__ __device__ __device_builtin__ double __CRTDECL hypot(double x, double y);
#else /* _WIN32 */
extern __host__ __device__ __device_builtin__ double           hypot(double x, double y) __THROW;
#endif /* _WIN32 */

#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate one over the square root of the sum of squares of two arguments.
 *
 * Calculate one over the length of the hypotenuse of a right triangle whose two sides have 
 * lengths \p x and \p y without undue overflow or underflow.
 *
 * \return Returns one over the length of the hypotenuse 
 * \latexonly $\frac{1}{\sqrt{x^2+y^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                rhypot(double x, double y) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculates the length of the hypotenuse of a right triangle whose two sides have lengths 
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse 
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
#if defined(_WIN32)
static __host__ __device__ __device_builtin__ float __CRTDECL hypotf(float x, float y);
#else /* _WIN32 */
extern __host__ __device__ __device_builtin__ float           hypotf(float x, float y) __THROW;
#endif /* _WIN32 */

#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate one over the square root of the sum of squares of two arguments.
 *
 * Calculates one over the length of the hypotenuse of a right triangle whose two sides have 
 * lengths \p x and \p y without undue overflow or underflow.
 *
 * \return Returns one over the length of the hypotenuse 
 * \latexonly $\frac{1}{\sqrt{x^2+y^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                 rhypotf(float x, float y) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculate the length of three dimensional vector \p p in euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of 3D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl norm3d(double a, double b, double c) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculate one over the length of three dimensional vector \p p in euclidean space undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vetor 
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                rnorm3d(double a, double b, double c) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculate the length of four dimensional vector \p p in euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of 4D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.t</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl norm4d(double a, double b, double c, double d) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculates the length of three dimensional vector \p p in euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of the 3D
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  norm3df(float a, float b, float c) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculates one over the length of three dimension vector \p p in euclidean space without undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vector
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                 rnorm3df(float a, float b, float c) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculates the length of four dimensional vector \p p in euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of the 4D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.t</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  norm4df(float a, float b, float c, float d) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x, 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 cbrt(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl cbrt(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x, 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrtf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  cbrtf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl cbrtf(float x);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return 
 * - rcbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rcbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 rcbrt(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return 
 * - rcbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rcbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  rcbrtf(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - sinpi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinpi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 sinpi(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - sinpif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinpif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  sinpif(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - cospi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cospi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 cospi(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return 
 * - cospif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cospif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  cospif(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief  Calculate the sine and cosine of the first input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians), 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinpi() and ::cospi().
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ void                   sincospi(double x, double *sptr, double *cptr);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief  Calculate the sine and cosine of the first input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians), 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinpif() and ::cospif().
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ void                   sincospif(float x, float *sptr, float *cptr);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y
 *
 * \return 
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - pow(-1, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - pow(+1, \p y) returns 1 for any \p y, even a NaN.
 * - pow(\p x, 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - pow(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - pow(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl pow(double x, double y) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The 
 * integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return 
 * - modf(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - modf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - modf(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl modf(double x, double *iptr) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * Calculate the floating-point remainder of \p x / \p y. 
 * The absolute value of the computed value is always less than \p y's
 * absolute value and will have the same sign as \p x.
 *
 * \return
 * - Returns the floating point remainder of \p x / \p y.
 * - fmod(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - fmod(\p x, \p y) returns NaN and raised an invalid floating point exception if \p x is 
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - fmod(\p x, \p y) returns zero if \p y is zero or the result would overflow.
 * - fmod(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - fmod(\p x, 0) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double         __cdecl fmod(double x, double y) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder.
 *
 * Compute double-precision floating-point remainder \p r of dividing 
 * \p x by \p y for nonzero \p y. Thus 
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * In the case when 
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return 
 * - remainder(\p x, 0) returns NaN.
 * - remainder(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - remainder(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 remainder(double x, double y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl remainder(double x, double y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute single-precision floating-point remainder.
 *
 * Compute single-precision floating-point remainder \p r of dividing 
 * \p x by \p y for nonzero \p y. Thus 
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * In the case when 
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return 
 * \return 
 * - remainderf(\p x, 0) returns NaN.
 * - remainderf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - remainderf(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  remainderf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl remainderf(float x, float y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the
 * ::remainder() function. Argument \p quo returns part of quotient upon 
 * division of \p x by \p y. Value \p quo has the same sign as 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return 
 * Returns the remainder.
 * - remquo(\p x, 0, \p quo) returns NaN.
 * - remquo(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - remquo(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 remquo(double x, double y, int *quo) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl remquo(double x, double y, int *quo);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute single-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the 
 * ::remainderf() function. Argument \p quo returns part of quotient upon 
 * division of \p x by \p y. Value \p quo has the same sign as 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return 
 * Returns the remainder.
 * - remquof(\p x, 0, \p quo) returns NaN.
 * - remquof(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - remquof(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  remquof(float x, float y, int *quo) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl remquof(float x, float y, int *quo);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x, 
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - j0(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl j0(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x, 
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - j0f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  j0f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x, 
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - j1(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl j1(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x, 
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - j1f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  j1f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x, 
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - jn(\p n, NaN) returns NaN.
 * - jn(\p n, \p x) returns NaN for \p n < 0.
 * - jn(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl jn(int n, double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x, 
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - jnf(\p n, NaN) returns NaN.
 * - jnf(\p n, \p x) returns NaN for \p n < 0.
 * - jnf(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  jnf(int n, float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x, 
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - y0(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y0(\p x) returns NaN for \p x < 0.
 * - y0(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl y0(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x, 
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - y0f(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y0f(\p x) returns NaN for \p x < 0.
 * - y0f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  y0f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x, 
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - y1(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y1(\p x) returns NaN for \p x < 0.
 * - y1(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl y1(double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x, 
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - y1f(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y1f(\p x) returns NaN for \p x < 0.
 * - y1f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  y1f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x, 
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - yn(\p n, \p x) returns NaN for \p n < 0.
 * - yn(\p n, 0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - yn(\p n, \p x) returns NaN for \p x < 0.
 * - yn(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - yn(\p n, NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl yn(int n, double x) __THROW;
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x, 
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - ynf(\p n, \p x) returns NaN for \p n < 0.
 * - ynf(\p n, 0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ynf(\p n, \p x) returns NaN for \p x < 0.
 * - ynf(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - ynf(\p n, NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  ynf(int n, float x) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x, 
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 0.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl cyl_bessel_i0(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x, 
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  cyl_bessel_i0f(float x) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x, 
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 1.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl cyl_bessel_i1(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x, 
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 1.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  cyl_bessel_i1f(float x) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- π --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 erf(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl erf(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- π --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return  
 * - erff(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  erff(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl erff(float x);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for 
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfinv(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfinv(-1) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 erfinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for 
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfinvf(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfinvf(-1) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  erfinvf(float y);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return 
 * - erfc(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - erfc(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 erfc(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl erfc(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return 
 * - erfcf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - erfcf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  erfcf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl erfcf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $\log_{e}\left|\int_{0}^{\infty} e^{-t}t^{x-1}dt\right|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mfenced open="|" close="|">
 *     <m:mrow>
 *       <m:msubsup>
 *         <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mn>0</m:mn>
 *         </m:mrow>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *         </m:mrow>
 *       </m:msubsup>
 *       <m:msup>
 *         <m:mi>e</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mi>t</m:mi>
 *         </m:mrow>
 *       </m:msup>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi>x</m:mi>
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mn>1</m:mn>
 *         </m:mrow>
 *       </m:msup>
 *       <m:mi>d</m:mi>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:mfenced>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \return 
 * - lgamma(1) returns +0.
 * - lgamma(2) returns +0.
 * - lgamma(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - lgamma(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 0.
 * - lgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - lgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 lgamma(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl lgamma(double x);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for 
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcinv(0) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfcinv(2) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 erfcinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for 
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcinvf(0) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfcinvf(2) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  erfcinvf(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval 
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdfinv(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinv(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinv(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 normcdfinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval 
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdfinvf(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinvf(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinvf(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  normcdfinvf(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - normcdf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 normcdf(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdff(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - normcdff(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0

 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  normcdff(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcx(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - erfcx(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - erfcx(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_double
 */
extern __host__ __device__ __device_builtin__ double                 erfcx(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcxf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - erfcxf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - erfcxf(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating point range.

 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  erfcxf(float x);

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $log_{e}|\ \int_{0}^{\infty} e^{-t}t^{x-1}dt|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mtext>&#xA0;</m:mtext>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - lgammaf(1) returns +0.
 * - lgammaf(2) returns +0.
 * - lgammaf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating point range.
 * - lgammaf(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  0.
 * - lgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - lgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  lgammaf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl lgammaf(float x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - tgamma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tgamma(2) returns +0.
 * - tgamma(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - tgamma(\p x) returns NaN if \p x < 0.
 * - tgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - tgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 tgamma(double x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl tgamma(double x);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - tgammaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tgammaf(2) returns +0.
 * - tgammaf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating point range.
 * - tgammaf(\p x) returns NaN if \p x < 0.
 * - tgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - tgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  tgammaf(float x) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl tgammaf(float x);
#endif /* _MSC_VER < 1800 */
/** \ingroup CUDA_MATH_DOUBLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 copysign(double x, double y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl copysign(double x, double y);
#endif /* _MSC_VER < 1800 */
/** \ingroup CUDA_MATH_SINGLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  copysignf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl copysignf(float x, float y);
#endif /* _MSC_VER < 1800 */
// FIXME exceptional cases for nextafter
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Return next representable double-precision floating-point value after argument.
 *
 * Calculate the next representable double-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafter()
 * returns the smallest representable number greater than \p x
 *
 * \return 
 * - nextafter(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 nextafter(double x, double y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl nextafter(double x, double y);
#endif /* _MSC_VER < 1800 */
// FIXME exceptional cases for nextafterf
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Return next representable single-precision floating-point value afer argument.
 *
 * Calculate the next representable single-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafterf()
 * returns the smallest representable number greater than \p x
 *
 * \return 
 * - nextafterf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  nextafterf(float x, float y) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl nextafterf(float x, float y);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Returns "Not a Number" value.
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return 
 * - nan(\p tagp) returns NaN.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 nan(const char *tagp) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl nan(const char *tagp);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Returns "Not a Number" value
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return 
 * - nanf(\p tagp) returns NaN.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  nanf(const char *tagp) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl nanf(const char *tagp);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* namespace std */
#endif
extern __host__ __device__ __device_builtin__ int                    __isinff(float) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isnanf(float) __THROW;


#if defined(__APPLE__)
extern __host__ __device__ __device_builtin__ int                    __isfinited(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isfinitef(float) __THROW;
extern __host__ __device__ __device_builtin__ int                    __signbitd(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isnand(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isinfd(double) __THROW;
#else /* __APPLE__ */
extern __host__ __device__ __device_builtin__ int                    __finite(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __finitef(float) __THROW;
extern __host__ __device__ __device_builtin__ int                    __signbit(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isnan(double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isinf(double) __THROW;
#endif /* __APPLE__ */

extern __host__ __device__ __device_builtin__ int                    __signbitf(float) __THROW;

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - fma(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fma(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fma(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ double                 fma(double x, double y, double z) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP double __cdecl fma(double x, double y, double z);
#endif /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - fmaf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if _MSC_VER < 1800
extern __host__ __device__ __device_builtin__ float                  fmaf(float x, float y, float z) __THROW;
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fmaf(float x, float y, float z);
#endif /* _MSC_VER < 1800 */
#ifdef __QNX__
} /* std */
#endif


/* these are here to avoid warnings on the call graph.
   long double is not supported on the device */
extern __host__ __device__ __device_builtin__ int                    __signbitl(long double) __THROW;
#if defined(__APPLE__)
extern __host__ __device__ __device_builtin__ int                    __isfinite(long double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isinf(long double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isnan(long double) __THROW;
#else /* __APPLE__ */
extern __host__ __device__ __device_builtin__ int                    __finitel(long double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isinfl(long double) __THROW;
extern __host__ __device__ __device_builtin__ int                    __isnanl(long double) __THROW;
#endif /* __APPLE__ */

#if defined(_WIN32) && defined(_M_AMD64)
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl acosf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl asinf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl atanf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl atan2f(float, float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl cosf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl sinf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl tanf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl coshf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl sinhf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl tanhf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl expf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl logf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl log10f(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl modff(float, float*) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl powf(float, float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl sqrtf(float) __THROW;         
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl ceilf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl floorf(float) __THROW;
extern _CRTIMP __host__ __device__ __device_builtin__ float __cdecl fmodf(float, float) __THROW;
#else /* _WIN32 && _M_AMD64 */

#ifdef __QNX__
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [0, 
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - acosf(1) returns +0.
 * - acosf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  acosf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - asinf(0) returns +0.
 * - asinf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  asinf(float x) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atanf(0) returns +0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  atanf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p y / \p x. The quadrant of the result is 
 * determined by the signs of inputs \p y and \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atan2f(0, 1) returns +0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  atan2f(float y, float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return 
 * - cosf(0) returns 1.
 * - cosf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ float                  cosf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return 
 * - sinf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ float                  sinf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return 
 * - tanf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tanf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ float                  tanf(float x) __THROW;
// FIXME return values for large arg values
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * - coshf(0) returns 1.
 * - coshf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  coshf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - sinhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinhf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  sinhf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - tanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  tanhf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the natural logarithm of the input argument.
 *
 * Calculate the natural logarithm of the input argument \p x.
 *
 * \return 
 * - logf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - logf(1) returns +0.
 * - logf(\p x) returns NaN for \p x < 0.
 * - logf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  logf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __host__ __device__ __device_builtin__ float                  expf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return 
 * - log10f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log10f(1) returns +0.
 * - log10f(\p x) returns NaN for \p x < 0.
 * - log10f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  log10f(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return 
 * - modff(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - modff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - modff(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  modff(float x, float *iptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y.
 *
 * \return 
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - powf(-1, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - powf(+1, \p y) returns 1 for any \p y, even a NaN.
 * - powf(\p x, 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - powf(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - powf(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  powf(float x, float y) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x, 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  sqrtf(float x) __THROW;         
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns 
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- ⌈ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- ⌉ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - ceilf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ceilf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
extern __host__ __device__ __device_builtin__ float                  ceilf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the largest integer less than or equal to \p x.
 * 
 * Calculate the largest integer value which is less than or equal to \p x.
 * 
 * \return
 * Returns 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - floorf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - floorf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  floorf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * Calculate the floating-point remainder of \p x / \p y. 
 * The absolute value of the computed value is always less than \p y's
 * absolute value and will have the same sign as \p x.
 *
 * \return
 * - Returns the floating point remainder of \p x / \p y.
 * - fmodf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - fmodf(\p x, \p y) returns NaN and raised an invalid floating point exception if \p x is 
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - fmodf(\p x, \p y) returns zero if \p y is zero or the result would overflow.
 * - fmodf(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - fmodf(\p x, 0) returns NaN.
 *
 * \note_accuracy_single
 */
extern __host__ __device__ __device_builtin__ float                  fmodf(float x, float y) __THROW;
#ifdef __QNX__
/* redeclare some builtins that QNX uses */
extern __host__ __device__ __device_builtin__ float _FLog(float, int);
extern __host__ __device__ __device_builtin__ float _FCosh(float, float);
extern __host__ __device__ __device_builtin__ float _FSinh(float, float);
extern __host__ __device__ __device_builtin__ float _FSinx(float, unsigned int, int);
extern __host__ __device__ __device_builtin__ int _FDsign(float);
extern __host__ __device__ __device_builtin__ int _Dsign(double);
} /* std */
#endif
#endif /* _WIN32 && _M_AMD64 */

#if !defined(__CUDACC_RTC__)
}
#endif /* !__CUDACC_RTC__ */

#if !defined(__CUDACC_RTC__)
#include <math.h>
#include <stdlib.h>

#ifndef __CUDA_INTERNAL_SKIP_CPP_HEADERS__
#include <cmath>
#include <cstdlib>
#endif /* __CUDA_INTERNAL_SKIP_CPP_HEADERS__ */
#endif /* __CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__)

__host__ __device__ __cudart_builtin__ int signbit(float x);
__host__ __device__ __cudart_builtin__ int signbit(double x);
__host__ __device__ __cudart_builtin__ int signbit(long double x);

__host__ __device__ __cudart_builtin__ int isfinite(float x);
__host__ __device__ __cudart_builtin__ int isfinite(double x);
__host__ __device__ __cudart_builtin__ int isfinite(long double x);

__host__ __device__ __cudart_builtin__ int isnan(float x);
__host__ __device__ __cudart_builtin__ int isnan(double x);
__host__ __device__ __cudart_builtin__ int isnan(long double x);

__host__ __device__ __cudart_builtin__ int isinf(float x);
__host__ __device__ __cudart_builtin__ int isinf(double x);
__host__ __device__ __cudart_builtin__ int isinf(long double x);

#elif defined(__GNUC__)

#undef signbit
#undef isfinite
#undef isnan
#undef isinf

#if defined(__APPLE__)

__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(float x);
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x);
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(float x); 
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x);
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(float x);
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x) throw();
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(float x);
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x) throw();
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(long double x);

#else /* __APPLE__ */

#ifdef __QNX__
/* QNX defines functions in std, need to declare them here,
 * redefine in CUDABE */
namespace std {
__host__ __device__ __cudart_builtin__ bool signbit(float x);
__host__ __device__ __cudart_builtin__ bool signbit(double x);
__host__ __device__ __cudart_builtin__ bool signbit(long double x);
}
#else /* !QNX */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(float x);
#if defined(__ICC)
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x) throw();
#else /* !__ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(double x);
#endif /* __ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int signbit(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(float x); 
#if defined(__ICC)
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x) throw();
#else /* !__ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(double x);
#endif /* __ICC */
__forceinline__ __host__ __device__ __cudart_builtin__ int isfinite(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(float x);
#if defined(__ANDROID__)
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x);
#else /* !__ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(double x) throw();
#endif /* __ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isnan(long double x);

__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(float x);
#if defined(__ANDROID__)
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x);
#else /* !__ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(double x) throw();
#endif /* __ANDROID__ */
__forceinline__ __host__ __device__ __cudart_builtin__ int isinf(long double x);
#endif /* QNX */

#endif /* __APPLE__ */

#if defined(__arm__) && !defined(_STLPORT_VERSION) && !_GLIBCXX_USE_C99
#if !defined(__ANDROID__) || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)

#ifdef __QNX__
/* QNX defines functions in std, need to declare them here,
 * redefine in CUDABE */
namespace std {
__host__ __device__ __cudart_builtin__ long long int abs (long long int a);
}
#else
static __inline__ __host__ __device__ __cudart_builtin__ long long int abs(long long int a);
#endif /* QNX */

#endif /* !__ANDROID__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8) */
#endif /* __arm__ && !_STLPORT_VERSION && !_GLIBCXX_USE_C99 */

#if (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)) && !defined(__ibmxl__)

#if !defined(_STLPORT_VERSION)
namespace __gnu_cxx
{
#endif /* !_STLPORT_VERSION */

extern __host__ __device__ __cudart_builtin__ long long int abs(long long int a);

#if !defined(_STLPORT_VERSION)
}
#endif /* !_STLPORT_VERSION */

#endif /* (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)) && !__ibmxl__ */

namespace std
{
  template<typename T> extern __host__ __device__ __cudart_builtin__ T __pow_helper(T, int);
  template<typename T> extern __host__ __device__ __cudart_builtin__ T __cmath_power(T, unsigned int);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;
using std::pow;
using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;

#elif defined(_WIN32)

extern __host__ __device__ __cudart_builtin__ _CRTIMP double __cdecl _hypot(double x, double y);
extern __host__ __device__ __cudart_builtin__ _CRTIMP float  __cdecl _hypotf(float x, float y);

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int signbit(long double a);
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __cudart_builtin__ bool signbit(long double);
extern __host__ __device__ __cudart_builtin__ __device_builtin__ _CRTIMP int _ldsign(long double);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE 
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is negative. 
 */
static __inline__ __host__ __device__ __RETURN_TYPE signbit(double a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE 
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is negative. 
 */
extern __host__ __device__ __cudart_builtin__ __RETURN_TYPE signbit(double);
extern __host__ __device__ __cudart_builtin__ __device_builtin__ _CRTIMP int _dsign(double);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'.  Returns a nonzero value 
 * if and only if \p a is negative.  
 */
static __inline__ __host__ __device__ __RETURN_TYPE signbit(float a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'.  Returns a nonzero value 
 * if and only if \p a is negative.  
 */
extern __host__ __device__ __cudart_builtin__ __RETURN_TYPE signbit(float);
extern __host__ __device__ __cudart_builtin__ __device_builtin__ _CRTIMP int _fdsign(float);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isinf(long double a);
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isinf(long double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 * \return
 * - With Visual Studio 2013 host compiler: Returns true if and only 
 * if \p a is a infinite value.
 * - With other host compilers: Returns a nonzero value if and only 
 * if \p a is a infinite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isinf(double a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 * \return
 * - With Visual Studio 2013 host compiler: Returns true if and only 
 * if \p a is a infinite value.
 * - With other host compilers: Returns a nonzero value if and only 
 * if \p a is a infinite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isinf(double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is a infinite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a nonzero 
 * value if and only if \p a is a infinite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isinf(float a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is a infinite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a nonzero 
 * value if and only if \p a is a infinite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isinf(float a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isnan(long double a);
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isnan(long double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isnan(double a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isnan(double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isnan(float a);
#else /* _MSC_VER < 1800 */
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool

static __inline__ __host__ __device__ __RETURN_TYPE isnan(float a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
static __inline__ __host__ __device__ int isfinite(long double a);
#else /* _MSC_VER < 1800 */
static __inline__ __host__ __device__ bool isfinite(long double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isfinite(double a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isfinite(double a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
#undef __RETURN_TYPE
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isfinite(float a);
#else /* _MSC_VER < 1800 */
#undef __RETURN_TYPE
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __host__ __device__ __RETURN_TYPE isfinite(float a);
#endif /* _MSC_VER < 1800 */

#if _MSC_VER < 1800
template<class T> extern __host__ __device__ __cudart_builtin__ T _Pow_int(T, int);
extern __host__ __device__ __cudart_builtin__ long long int abs(long long int);
#else /* _MSC_VER < 1800 */
template<class T> extern __host__ __device__ __cudart_builtin__ T _Pow_int(T, int) throw();
extern __host__ __device__ __cudart_builtin__ long long int abs(long long int) throw();
#endif /* _MSC_VER < 1800 */

#endif /* __CUDACC_RTC__ */

#if defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-extensions"
#endif /* __clang__ */
_LIBCPP_BEGIN_NAMESPACE_STD
#elif defined(__GNUC__) && !defined(_STLPORT_VERSION)
namespace std {
#endif /* defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION) ||
          __GNUC__ && !_STLPORT_VERSION */

#if defined(__CUDACC_RTC__) || defined(__GNUC__)

#if defined(__CUDACC_RTC__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || defined(__ibmxl__)
extern __host__ __device__ __cudart_builtin__ long long int abs(long long int);
#endif /* __CUDACC__JIT__ || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __ibmxl__ */

#endif /* __CUDACC_RTC__ || __GNUC__ */

#if defined(__CUDACC_RTC__) || _MSC_VER < 1800 && (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101))
extern __host__ __device__ __cudart_builtin__ long int __cdecl abs(long int);
extern __host__ __device__ __cudart_builtin__ float    __cdecl abs(float);
extern __host__ __device__ __cudart_builtin__ double   __cdecl abs(double);
extern __host__ __device__ __cudart_builtin__ float    __cdecl fabs(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl ceil(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl floor(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl sqrt(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl pow(float, float);
#if !defined(__QNX__) && !(defined(__GNUC__) && __cplusplus >= 201103L)
/* provide in math_functions.hpp */
extern __host__ __device__ __cudart_builtin__ float    __cdecl pow(float, int);
extern __host__ __device__ __cudart_builtin__ double   __cdecl pow(double, int);
#endif  /* !defined(__QNX__) && !(defined(__GNUC__) && __cplusplus >= 201103L)  */
extern __host__ __device__ __cudart_builtin__ float    __cdecl log(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl log10(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl fmod(float, float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl modf(float, float*);
extern __host__ __device__ __cudart_builtin__ float    __cdecl exp(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl frexp(float, int*);
extern __host__ __device__ __cudart_builtin__ float    __cdecl ldexp(float, int);
extern __host__ __device__ __cudart_builtin__ float    __cdecl asin(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl sin(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl sinh(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl acos(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl cos(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl cosh(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl atan(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl atan2(float, float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl tan(float);
extern __host__ __device__ __cudart_builtin__ float    __cdecl tanh(float);
#else /* __CUDACC_RTC__ || _MSC_VER < 1800 && (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101)) */
extern __host__ __device__ __cudart_builtin__ long int __cdecl abs(long int) throw();
#if defined(_LIBCPP_VERSION)
extern __host__ __device__ __cudart_builtin__ long long int __cdecl abs(long long int) throw();
#endif /* defined(_LIBCPP_VERSION) */
extern __host__ __device__ __cudart_builtin__ float    __cdecl abs(float) throw();
extern __host__ __device__ __cudart_builtin__ double   __cdecl abs(double) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl fabs(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl ceil(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl floor(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl sqrt(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl pow(float, float) throw();
#if !(defined(__GNUC__) && __cplusplus >= 201103L)
extern __host__ __device__ __cudart_builtin__ float    __cdecl pow(float, int) throw();
extern __host__ __device__ __cudart_builtin__ double   __cdecl pow(double, int) throw();
#endif /* !(defined(__GNUC__) && __cplusplus >= 201103L) */
extern __host__ __device__ __cudart_builtin__ float    __cdecl log(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl log10(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl fmod(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl modf(float, float*) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl exp(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl frexp(float, int*) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl ldexp(float, int) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl asin(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl sin(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl sinh(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl acos(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl cos(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl cosh(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl atan(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl atan2(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl tan(float) throw();
extern __host__ __device__ __cudart_builtin__ float    __cdecl tanh(float) throw();
#endif /* __CUDACC_RTC__ || _MSC_VER < 1800 && (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101)) */

#if defined(_LIBCPP_VERSION) && defined(_LIBCPP_END_NAMESPACE_STD) && !defined(_STLPORT_VERSION)
_LIBCPP_END_NAMESPACE_STD
#if defined(__clang__)
#pragma clang diagnostic pop
#endif /* __clang__ */
#elif defined(__GNUC__) && !defined(_STLPORT_VERSION)
}
#endif /* defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION) ||
          __GNUC__ && !_STLPORT_VERSION */

#if defined(__CUDACC_RTC__)
#define __MATH_FUNCTIONS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline __host__ __device__
#endif /* __CUDACC_RTC__ */

#if _MSC_VER < 1800
#ifdef __QNX__
namespace std {
__host__ __device__ __cudart_builtin__ float logb(float a);
__host__ __device__ __cudart_builtin__ int ilogb(float a);
__host__ __device__ __cudart_builtin__ int ilogbf(float a);

__host__ __device__ __cudart_builtin__ float scalbn(float a, int b);
__host__ __device__ __cudart_builtin__ float scalbln(float a, long int b);
__host__ __device__ __cudart_builtin__ float exp2(float a);
__host__ __device__ __cudart_builtin__ float expm1(float a);
__host__ __device__ __cudart_builtin__ float log2(float a);
__host__ __device__ __cudart_builtin__ float log1p(float a);
__host__ __device__ __cudart_builtin__ float acosh(float a);
__host__ __device__ __cudart_builtin__ float asinh(float a);
__host__ __device__ __cudart_builtin__ float atanh(float a);
__host__ __device__ __cudart_builtin__ float hypot(float a, float b);
__host__ __device__ __cudart_builtin__ float norm3d(float a, float b, float c);
__host__ __device__ __cudart_builtin__ float norm4d(float a, float b, float c, float d);
__host__ __device__ __cudart_builtin__ float cbrt(float a);
__host__ __device__ __cudart_builtin__ float erf(float a);
__host__ __device__ __cudart_builtin__ float erfc(float a);
__host__ __device__ __cudart_builtin__ float lgamma(float a);
__host__ __device__ __cudart_builtin__ float tgamma(float a);
__host__ __device__ __cudart_builtin__ float copysign(float a, float b);
__host__ __device__ __cudart_builtin__ float nextafter(float a, float b);
__host__ __device__ __cudart_builtin__ float remainder(float a, float b);
__host__ __device__ __cudart_builtin__ float remquo(float a, float b, int *quo);
__host__ __device__ __cudart_builtin__ float round(float a);
__host__ __device__ __cudart_builtin__ long int lround(float a);
__host__ __device__ __cudart_builtin__ long long int llround(float a);
__host__ __device__ __cudart_builtin__ float trunc(float a);
__host__ __device__ __cudart_builtin__ float rint(float a);
__host__ __device__ __cudart_builtin__ long int lrint(float a);
__host__ __device__ __cudart_builtin__ long long int llrint(float a);
__host__ __device__ __cudart_builtin__ float nearbyint(float a);
__host__ __device__ __cudart_builtin__ float fdim(float a, float b);
__host__ __device__ __cudart_builtin__ float fma(float a, float b, float c);
__host__ __device__ __cudart_builtin__ float fmax(float a, float b);
__host__ __device__ __cudart_builtin__ float fmin(float a, float b);
}
#else /* !QNX */
__MATH_FUNCTIONS_DECL__ float logb(float a);

__MATH_FUNCTIONS_DECL__ int ilogb(float a);

__MATH_FUNCTIONS_DECL__ float scalbn(float a, int b);

__MATH_FUNCTIONS_DECL__ float scalbln(float a, long int b);

__MATH_FUNCTIONS_DECL__ float exp2(float a);

__MATH_FUNCTIONS_DECL__ float expm1(float a);

__MATH_FUNCTIONS_DECL__ float log2(float a);

__MATH_FUNCTIONS_DECL__ float log1p(float a);

__MATH_FUNCTIONS_DECL__ float acosh(float a);

__MATH_FUNCTIONS_DECL__ float asinh(float a);

__MATH_FUNCTIONS_DECL__ float atanh(float a);

__MATH_FUNCTIONS_DECL__ float hypot(float a, float b);

__MATH_FUNCTIONS_DECL__ float norm3d(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float norm4d(float a, float b, float c, float d);

__MATH_FUNCTIONS_DECL__ float cbrt(float a);

__MATH_FUNCTIONS_DECL__ float erf(float a);

__MATH_FUNCTIONS_DECL__ float erfc(float a);

__MATH_FUNCTIONS_DECL__ float lgamma(float a);

__MATH_FUNCTIONS_DECL__ float tgamma(float a);

__MATH_FUNCTIONS_DECL__ float copysign(float a, float b);

__MATH_FUNCTIONS_DECL__ float nextafter(float a, float b);

__MATH_FUNCTIONS_DECL__ float remainder(float a, float b);

__MATH_FUNCTIONS_DECL__ float remquo(float a, float b, int *quo);

__MATH_FUNCTIONS_DECL__ float round(float a);

__MATH_FUNCTIONS_DECL__ long int lround(float a);

__MATH_FUNCTIONS_DECL__ long long int llround(float a);

__MATH_FUNCTIONS_DECL__ float trunc(float a);

__MATH_FUNCTIONS_DECL__ float rint(float a);

__MATH_FUNCTIONS_DECL__ long int lrint(float a);

__MATH_FUNCTIONS_DECL__ long long int llrint(float a);

__MATH_FUNCTIONS_DECL__ float nearbyint(float a);

__MATH_FUNCTIONS_DECL__ float fdim(float a, float b);

__MATH_FUNCTIONS_DECL__ float fma(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float fmax(float a, float b);

__MATH_FUNCTIONS_DECL__ float fmin(float a, float b);
#endif /* QNX */
#else /* _MSC_VER < 1800 */
extern __host__ __device__ __cudart_builtin__ float __cdecl logb(float) throw();
extern __host__ __device__ __cudart_builtin__ int   __cdecl ilogb(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl scalbn(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl scalbln(float, long int) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl exp2(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl expm1(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl log2(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl log1p(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl acosh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl asinh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl atanh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl hypot(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl norm3d(float, float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl norm4d(float, float, float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl cbrt(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl erf(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl erfc(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl lgamma(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl tgamma(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl copysign(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl nextafter(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl remainder(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl remquo(float, float, int *) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl round(float) throw();
extern __host__ __device__ __cudart_builtin__ long int      __cdecl lround(float) throw();
extern __host__ __device__ __cudart_builtin__ long long int __cdecl llround(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl trunc(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl rint(float) throw();
extern __host__ __device__ __cudart_builtin__ long int      __cdecl lrint(float) throw();
extern __host__ __device__ __cudart_builtin__ long long int __cdecl llrint(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl nearbyint(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fdim(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fma(float, float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fmax(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fmin(float, float) throw();
#endif /* _MSC_VER < 1800 */

__MATH_FUNCTIONS_DECL__ float exp10(float a);

__MATH_FUNCTIONS_DECL__ float rsqrt(float a);

__MATH_FUNCTIONS_DECL__ float rcbrt(float a);

__MATH_FUNCTIONS_DECL__ float sinpi(float a);

__MATH_FUNCTIONS_DECL__ float cospi(float a);

__MATH_FUNCTIONS_DECL__ void sincospi(float a, float *sptr, float *cptr);

__MATH_FUNCTIONS_DECL__ void sincos(float a, float *sptr, float *cptr);

__MATH_FUNCTIONS_DECL__ float j0(float a);

__MATH_FUNCTIONS_DECL__ float j1(float a);

__MATH_FUNCTIONS_DECL__ float jn(int n, float a);

__MATH_FUNCTIONS_DECL__ float y0(float a);

__MATH_FUNCTIONS_DECL__ float y1(float a);

__MATH_FUNCTIONS_DECL__ float yn(int n, float a);

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i0(float a);

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i1(float a);

__MATH_FUNCTIONS_DECL__ float erfinv(float a);

__MATH_FUNCTIONS_DECL__ float erfcinv(float a);

__MATH_FUNCTIONS_DECL__ float normcdfinv(float a);

__MATH_FUNCTIONS_DECL__ float normcdf(float a);

__MATH_FUNCTIONS_DECL__ float erfcx(float a);

__MATH_FUNCTIONS_DECL__ double copysign(double a, float b);

__MATH_FUNCTIONS_DECL__ float copysign(float a, double b);

__MATH_FUNCTIONS_DECL__ unsigned int min(unsigned int a, unsigned int b);

__MATH_FUNCTIONS_DECL__ unsigned int min(int a, unsigned int b);

__MATH_FUNCTIONS_DECL__ unsigned int min(unsigned int a, int b);

__MATH_FUNCTIONS_DECL__ long long int min(long long int a, long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int min(unsigned long long int a, unsigned long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int min(long long int a, unsigned long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int min(unsigned long long int a, long long int b);

__MATH_FUNCTIONS_DECL__ float min(float a, float b);

__MATH_FUNCTIONS_DECL__ double min(double a, double b);

__MATH_FUNCTIONS_DECL__ double min(float a, double b);

__MATH_FUNCTIONS_DECL__ double min(double a, float b);

__MATH_FUNCTIONS_DECL__ unsigned int max(unsigned int a, unsigned int b);

__MATH_FUNCTIONS_DECL__ unsigned int max(int a, unsigned int b);

__MATH_FUNCTIONS_DECL__ unsigned int max(unsigned int a, int b);

__MATH_FUNCTIONS_DECL__ long long int max(long long int a, long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int max(unsigned long long int a, unsigned long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int max(long long int a, unsigned long long int b);

__MATH_FUNCTIONS_DECL__ unsigned long long int max(unsigned long long int a, long long int b);

__MATH_FUNCTIONS_DECL__ float max(float a, float b);

__MATH_FUNCTIONS_DECL__ double max(double a, double b);

__MATH_FUNCTIONS_DECL__ double max(float a, double b);

__MATH_FUNCTIONS_DECL__ double max(double a, float b);

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

__MATH_FUNCTIONS_DECL__ float rintf(float a);

__MATH_FUNCTIONS_DECL__ long int lrintf(float a);

__MATH_FUNCTIONS_DECL__ long long int llrintf(float a);

__MATH_FUNCTIONS_DECL__ float nearbyintf(float a);

__MATH_FUNCTIONS_DECL__ int __signbitf(float a);

#if _MSC_VER >= 1800

__MATH_FUNCTIONS_DECL__ int __signbitl(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DECL__ int _ldsign(/* we do not support long double yet, hence double */double a);

__MATH_FUNCTIONS_DECL__ int __signbit(double a);
__MATH_FUNCTIONS_DECL__ int _dsign(double a);


__MATH_FUNCTIONS_DECL__ __forceinline__ int _fdsign(float a);

#endif

__MATH_FUNCTIONS_DECL__ float copysignf(float a, float b);

__MATH_FUNCTIONS_DECL__ int __finitef(float a);

#if defined(__APPLE__)

__MATH_FUNCTIONS_DECL__ int __isfinitef(float a);

#endif /* __APPLE__ */

__MATH_FUNCTIONS_DECL__ int __isinff(float a);

__MATH_FUNCTIONS_DECL__ int __isnanf(float a);

__MATH_FUNCTIONS_DECL__ float nextafterf(float a, float b);

__MATH_FUNCTIONS_DECL__ float nanf(const char *tagp);

__MATH_FUNCTIONS_DECL__ float sinf(float a);

__MATH_FUNCTIONS_DECL__ float cosf(float a);

__MATH_FUNCTIONS_DECL__ void sincosf(float a, float *sptr, float *cptr);

__MATH_FUNCTIONS_DECL__ float sinpif(float a);

__MATH_FUNCTIONS_DECL__ float cospif(float a);

__MATH_FUNCTIONS_DECL__ void sincospif(float a, float *sptr, float *cptr);

__MATH_FUNCTIONS_DECL__ float tanf(float a);

__MATH_FUNCTIONS_DECL__ float log2f(float a);

__MATH_FUNCTIONS_DECL__ float expf(float a);

__MATH_FUNCTIONS_DECL__ float exp10f(float a);

__MATH_FUNCTIONS_DECL__ float coshf(float a);

__MATH_FUNCTIONS_DECL__ float sinhf(float a);

__MATH_FUNCTIONS_DECL__ float tanhf(float a);

__MATH_FUNCTIONS_DECL__ float atan2f(float a, float b);

__MATH_FUNCTIONS_DECL__ float atanf(float a);

__MATH_FUNCTIONS_DECL__ float asinf(float a);

__MATH_FUNCTIONS_DECL__ float acosf(float a);

__MATH_FUNCTIONS_DECL__ float logf(float a);

__MATH_FUNCTIONS_DECL__ float log10f(float a);

__MATH_FUNCTIONS_DECL__ float log1pf(float a);

__MATH_FUNCTIONS_DECL__ float acoshf(float a);

__MATH_FUNCTIONS_DECL__ float asinhf(float a);

__MATH_FUNCTIONS_DECL__ float atanhf(float a);

__MATH_FUNCTIONS_DECL__ float expm1f(float a);

__MATH_FUNCTIONS_DECL__ float hypotf(float a, float b);

__MATH_FUNCTIONS_DECL__ float rhypotf(float a, float b);

__MATH_FUNCTIONS_DECL__ float norm3df(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float rnorm3df(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float norm4df(float a, float b, float c, float d);

__MATH_FUNCTIONS_DECL__ float cbrtf(float a);

__MATH_FUNCTIONS_DECL__ float rcbrtf(float a);

__MATH_FUNCTIONS_DECL__ float j0f(float a);

__MATH_FUNCTIONS_DECL__ float j1f(float a);

__MATH_FUNCTIONS_DECL__ float y0f(float a);

__MATH_FUNCTIONS_DECL__ float y1f(float a);

__MATH_FUNCTIONS_DECL__ float ynf(int n, float a);

__MATH_FUNCTIONS_DECL__ float jnf(int n, float a);

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i0f(float a);

__MATH_FUNCTIONS_DECL__ float cyl_bessel_i1f(float a);

__MATH_FUNCTIONS_DECL__ float erff(float a);

__MATH_FUNCTIONS_DECL__ float erfinvf(float a);

__MATH_FUNCTIONS_DECL__ float erfcf(float a);

__MATH_FUNCTIONS_DECL__ float erfcxf(float a);

__MATH_FUNCTIONS_DECL__ float erfcinvf(float a);

__MATH_FUNCTIONS_DECL__ float normcdfinvf(float a);

__MATH_FUNCTIONS_DECL__ float normcdff(float a);

__MATH_FUNCTIONS_DECL__ float lgammaf(float a);

__MATH_FUNCTIONS_DECL__ float ldexpf(float a, int b);

__MATH_FUNCTIONS_DECL__ float scalbnf(float a, int b);

__MATH_FUNCTIONS_DECL__ float scalblnf(float a, long int b);

__MATH_FUNCTIONS_DECL__ float frexpf(float a, int *b);

__MATH_FUNCTIONS_DECL__ float modff(float a, float *b);

__MATH_FUNCTIONS_DECL__ float fmodf(float a, float b);

__MATH_FUNCTIONS_DECL__ float remainderf(float a, float b);

__MATH_FUNCTIONS_DECL__ float remquof(float a, float b, int* quo);

__MATH_FUNCTIONS_DECL__ float fmaf(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float powif(float a, int b);

__MATH_FUNCTIONS_DECL__ double powi(double a, int b);

__MATH_FUNCTIONS_DECL__ float powf(float a, float b);

__MATH_FUNCTIONS_DECL__ float tgammaf(float a);

__MATH_FUNCTIONS_DECL__ float roundf(float a);

__MATH_FUNCTIONS_DECL__ long long int llroundf(float a);

__MATH_FUNCTIONS_DECL__ long int lroundf(float a);

__MATH_FUNCTIONS_DECL__ float fdimf(float a, float b);

__MATH_FUNCTIONS_DECL__ int ilogbf(float a);

__MATH_FUNCTIONS_DECL__ float logbf(float a);

#ifdef __QNX__
/* provide own versions of QNX builtins */
__MATH_FUNCTIONS_DECL__ float _FLog(float a, int tag);
__MATH_FUNCTIONS_DECL__ float _FCosh (float a, float b);
__MATH_FUNCTIONS_DECL__ float _FSinh (float a, float b);
__MATH_FUNCTIONS_DECL__ float _FSinx (float a, unsigned int tag, int c);
__MATH_FUNCTIONS_DECL__ int _FDsign (float a);
__MATH_FUNCTIONS_DECL__ int _Dsign (double a);
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

__func__(double rsqrt(double a));

__func__(double rcbrt(double a));

__func__(double sinpi(double a));

__func__(double cospi(double a));

__func__(void sincospi(double a, double *sptr, double *cptr));

__func__(double erfinv(double a));

__func__(double erfcinv(double a));

__func__(double normcdfinv(double a));

__func__(double normcdf(double a));

__func__(double erfcx(double a));

__func__(float rsqrtf(float a));

__func__(float rcbrtf(float a));

__func__(float sinpif(float a));

__func__(float cospif(float a));

__func__(void sincospif(float a, float *sptr, float *cptr));

__func__(float erfinvf(float a));

__func__(float erfcinvf(float a));

__func__(float normcdfinvf(float a));

__func__(float normcdff(float a));

__func__(float erfcxf(float a));

__func__(int min(int a, int b));

__func__(unsigned int umin(unsigned int a, unsigned int b));

__func__(long long int llmin(long long int a, long long int b));

__func__(unsigned long long int ullmin(unsigned long long int a, unsigned long long int b));

__func__(int max(int a, int b));

__func__(unsigned int umax(unsigned int a, unsigned int b));

__func__(long long int llmax(long long int a, long long int b));

__func__(unsigned long long int ullmax(unsigned long long int a, unsigned long long int b));

#if defined(_WIN32) || defined(__APPLE__) || defined (__ANDROID__)

__func__(int __isnan(double a));

#endif /* _WIN32 || __APPLE__ || __ANDROID__ */

#if defined(_WIN32) || defined(__APPLE__) || defined (__QNX__)

__func__(void sincos(double a, double *sptr, double *cptr));

#endif /* _WIN32 || __APPLE__ || __QNX__ */

#if defined(_WIN32) || defined(__APPLE__)

__func__(double exp10(double a));

__func__(float exp10f(float a));

__func__(void sincosf(float a, float *sptr, float *cptr));

__func__(int __isinf(double a));

#endif /* _WIN32 || __APPLE__ */

#if (defined(_WIN32) && _MSC_VER < 1800) || defined (__ANDROID__)

__func__(double log2(double a));

#endif /* (_WIN32 && _MSC_VER < 1800) || __ANDROID__ */

#if defined(_WIN32)

__func__(int __signbit(double a));

__func__(int __finite(double a));

__func__(int __signbitl(long double a));

__func__(int __signbitf(float a));

__func__(int __finitel(long double a));

__func__(int __finitef(float a));

__func__(int __isinfl(long double a));

__func__(int __isinff(float a));

__func__(int __isnanl(long double a));

__func__(int __isnanf(float a));

#endif /* _WIN32 */

#if defined(_WIN32) && _MSC_VER < 1800

__func__(double copysign(double a, double b));

__func__(double fmax(double a, double b));

__func__(double fmin(double a, double b));

__func__(double trunc(double a));

__func__(double round(double a));

__func__(long int lround(double a));

__func__(long long int llround(double a));

__func__(double rint(double a));

__func__(double nearbyint(double a));

__func__(long int lrint(double a));

__func__(long long int llrint(double a));

__func__(double fdim(double a, double b));

__func__(double scalbn(double a, int b));

__func__(double scalbln(double a, long int b));

__func__(double exp2(double a));

__func__(double log1p(double a));

__func__(double expm1(double a));

__func__(double cbrt(double a));

__func__(double acosh(double a));

__func__(double asinh(double a));

__func__(double atanh(double a));

__func__(int ilogb(double a));

__func__(double logb(double a));

__func__(double remquo(double a, double b, int *quo));

__func__(double remainder(double a, double b));

__func__(double fma (double a, double b, double c));

__func__(double nextafter(double a, double b));

__func__(double erf(double a));

__func__(double erfc(double a));

__func__(double lgamma(double a));

__func__(unsigned long long int __internal_host_nan_kernel(const char *s));

__func__(double nan(const char *tagp));

__func__(double __host_tgamma_kernel(double a));

__func__(double __host_stirling_poly(double a));

__func__(double __host_tgamma_stirling(double a));

__func__(double tgamma(double a));

__func__(float fmaxf(float a, float b));

__func__(float fminf(float a, float b));

__func__(float roundf(float a));

__func__(long int lroundf(float a));

__func__(long long int llroundf(float a));

__func__(float truncf(float a));

__func__(float rintf(float a));

__func__(float nearbyintf(float a));

__func__(long int lrintf(float a));

__func__(long long int llrintf(float a));

__func__(float logbf(float a));

__func__(float scalblnf(float a, long int b));

__func__(float log2f(float a));

__func__(float exp2f(float a));

__func__(float acoshf(float a));

__func__(float asinhf(float a));

__func__(float atanhf(float a));

__func__(float cbrtf(float a));

__func__(float expm1f(float a));

__func__(float fdimf(float a, float b));

__func__(float log1pf(float a));

__func__(float scalbnf(float a, int b));

__func__(float fmaf(float a, float b, float c));

__func__(int ilogbf(float a));

__func__(float erff(float a));

__func__(float erfcf(float a));

__func__(float lgammaf(float a));

__func__(float tgammaf(float a));

__func__(float remquof(float a, float b, int *quo));

__func__(float remainderf(float a, float b));

__func__(float copysignf(float a, float b));

__func__(float nextafterf(float a, float b));

__func__(float nanf(const char *tagp));

#endif /* _WIN32 && _MSC_VER < 1800 */

#if defined(_WIN32)

#pragma warning(default: 4211)

#endif /* _WIN32 */

#endif /* __CUDACC_RTC__ || __CUDABE__ */

#endif /* __CUDACC_RTC__ || !__CUDACC__ */

#if !defined(__CUDACC_RTC__)

#include "math_functions.hpp"

#endif /* !__CUDACC_RTC__ */

#include "math_functions_dbl_ptx3.h"

#endif /* !__MATH_FUNCTIONS_H__ */
