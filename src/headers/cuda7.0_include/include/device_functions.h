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

#if !defined(__DEVICE_FUNCTIONS_H__)
#define __DEVICE_FUNCTIONS_H__

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

extern "C"
{
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit integers.
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __mulhi(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit unsigned integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit unsigned integers. 
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __umulhi(unsigned int x, unsigned int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit integers. 
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __mul64hi(long long int x, long long int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit unsigned integers. 
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in an integer as a float.
 *
 * Reinterpret the bits in the signed integer value \p x as a single-precision
 * floating point value.
 * \return Returns reinterpreted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __int_as_float(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a float as a signed integer.
 *
 * Reinterpret the bits in the single-precision floating point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __float_as_int(float x);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __syncthreads(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __prof_trigger(int);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __threadfence(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __threadfence_block(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __trap(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   __brkpt(int c = 0);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Clamp the input argument to [+0.0, 1.0].
 *
 * Clamp the input argument \p x to be within the interval [+0.0, 1.0].
 * \return 
 * - __saturatef(\p x) returns 0 if \p x < 0.
 * - __saturatef(\p x) returns 1 if \p x > 1.
 * - __saturatef(\p x) returns \p x if 
 * \latexonly $0 \le x \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __saturatef(NaN) returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __saturatef(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the sum of absolute difference.
 *
 * Calculate 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the 32-bit sum of the third argument \p z plus and the absolute 
 * value of the difference between the first argument, \p x, and second 
 * argument, \p y.
 * 
 * Inputs \p x and \p y are signed 32-bit integers, input \p z is 
 * a 32-bit unsigned integer.
 *
 * \return Returns 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __sad(int x, int y, unsigned int z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the sum of absolute difference.
 *
 * Calculate 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the 32-bit sum of the third argument \p z plus and the absolute 
 * value of the difference between the first argument, \p x, and second 
 * argument, \p y.
 * 
 * Inputs \p x, \p y, and \p z are unsigned 32-bit integers.
 * 
 * \return Returns 
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __usad(unsigned int x, unsigned int y, unsigned int z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of \p x and \p y are ignored.
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __mul24(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of  \p x and  \p y are ignored. 
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __umul24(unsigned int x, unsigned int y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Divide two floating point values.
 *
 * Compute \p x divided by \p y.  If <tt>--use_fast_math</tt> is specified,
 * use ::__fdividef() for higher performance, otherwise use normal division.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fdividef(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate division of the input arguments.
 *
 * Calculate the fast approximate division of \p x by \p y.
 *
 * \return Returns \p x / \p y.
 * - __fdividef(
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN for 
 * \latexonly $2^{126} < y < 2^{128}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>126</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&lt;</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&lt;</m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>128</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __fdividef(\p x, \p y) returns 0 for 
 * \latexonly $2^{126} < y < 2^{128}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>126</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&lt;</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&lt;</m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>128</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and 
 * \latexonly $x \ne \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2260;<!-- ≠ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fdividef(float x, float y);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 fdivide(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate sine of the input argument.
 *
 * Calculate the fast approximate sine of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate sine of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __sinf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate cosine of the input argument.
 *
 * Calculate the fast approximate cosine of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate cosine of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __cosf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate tangent of the input argument.
 *
 * Calculate the fast approximate tangent of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate tangent of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note The result is computed as the fast divide of ::__sinf()
 * by ::__cosf(). Denormal input and output are flushed to sign-preserving 
 * 0.0 at each step of the computation.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __tanf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate of sine and cosine of the first input argument.
 *
 * Calculate the fast approximate of sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the second 
 * argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return
 * - none
 *
 * \note_accuracy_single_intrinsic
 * \note Denorm input/output is flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ void                   __sincosf(float x, float *sptr, float *cptr) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 
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
 * Calculate the fast approximate base 
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
 * \return Returns an approximation to 
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
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __expf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 10 exponential of the input argument.
 *
 * Calculate the fast approximate base 10 exponential of the input argument \p x, 
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
 * \return Returns an approximation to 
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
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __exp10f(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 2 logarithm of the input argument.
 *
 * Calculate the fast approximate base 2 logarithm of the input argument \p x.
 *
 * \return Returns an approximation to 
 * \latexonly $\log_2(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mn>2</m:mn>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __log2f(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 10 logarithm of the input argument.
 *
 * Calculate the fast approximate base 10 logarithm of the input argument \p x.
 *
 * \return Returns an approximation to 
 * \latexonly $\log_{10}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>10</m:mn>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __log10f(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 
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
 * Calculate the fast approximate base 
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
 * \return Returns an approximation to 
 * \latexonly $\log_e(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mi>e</m:mi>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __logf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate of 
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the fast approximate of \p x, the first input argument, 
 * raised to the power of \p y, the second input argument, 
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns an approximation to 
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ float                  __powf(float x, float y) __THROW;
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __float2int_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __float2int_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __float2int_ru(float);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __float2int_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __float2uint_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __float2uint_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __float2uint_ru(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __float2uint_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __int2float_rn(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __int2float_rz(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __int2float_ru(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __int2float_rd(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __uint2float_rn(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __uint2float_rz(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __uint2float_ru(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __uint2float_rd(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __float2ll_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __float2ll_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __float2ll_ru(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __float2ll_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __float2ull_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-towards_zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __float2ull_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __float2ull_ru(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __float2ull_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ll2float_rn(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ll2float_rz(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ll2float_ru(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ll2float_rd(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ull2float_rn(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ull2float_rz(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ull2float_ru(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __ull2float_rd(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a single-precision float to a half-precision float in round-to-nearest-even mode.
 *
 * Convert the single-precision float value \p x to a half-precision floating point value
 * represented in <tt>unsigned short</tt> format, in round-to-nearest-even mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned short         __float2half_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a half-precision float to a single-precision float in round-to-nearest-even mode.
 *
 * Convert the half-precision floating point value \p x represented in
 * <tt>unsigned short</tt> format to a single-precision floating point value.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __half2float(unsigned short x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-to-nearest-even mode.
 * 
 * Compute the sum of \p x and \p y in round-to-nearest-even rounding mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fadd_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-towards-zero mode.
 * 
 * Compute the sum of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fadd_rz(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-up mode.
 * 
 * Compute the sum of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fadd_ru(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-down mode.
 * 
 * Compute the sum of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fadd_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-to-nearest-even mode.
 * 
 * Compute the difference of \p x and \p y in round-to-nearest-even rounding mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsub_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-towards-zero mode.
 * 
 * Compute the difference of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsub_rz(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-up mode.
 * 
 * Compute the difference of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsub_ru(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-down mode.
 * 
 * Compute the difference of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsub_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-to-nearest-even mode.
 * 
 * Compute the product of \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmul_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-towards-zero mode.
 * 
 * Compute the product of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmul_rz(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-up mode.
 * 
 * Compute the product of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmul_ru(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-down mode.
 * 
 * Compute the product of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmul_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
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
 *  as a single operation, in round-to-nearest-even mode.
 * 
 * Computes the value of 
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
 *  as a single ternary operation, rounding the
 * result once in round-to-nearest-even mode.
 *
 * \return Returns the rounded value of 
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
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmaf_rn(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
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
 *  as a single operation, in round-towards-zero mode.
 * 
 * Computes the value of 
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
 *  as a single ternary operation, rounding the
 * result once in round-towards-zero mode.
 *
 * \return Returns the rounded value of 
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
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmaf_rz(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
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
 *  as a single operation, in round-up mode.
 * 
 * Computes the value of 
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
 *  as a single ternary operation, rounding the
 * result once in round-up (to positive infinity) mode.
 *
 * \return Returns the rounded value of 
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
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmaf_ru(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
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
 *  as a single operation, in round-down mode.
 * 
 * Computes the value of 
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
 *  as a single ternary operation, rounding the
 * result once in round-down (to negative infinity) mode.
 *
 * \return Returns the rounded value of 
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
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fmaf_rd(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 * 
 * Compute the reciprocal of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __frcp_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 * 
 * Compute the reciprocal of \p x in round-towards-zero mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __frcp_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 * 
 * Compute the reciprocal of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __frcp_ru(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 * 
 * Compute the reciprocal of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __frcp_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 * 
 * Compute the square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
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
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsqrt_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 * 
 * Compute the square root of \p x in round-towards-zero mode.
 *
 * \return Returns 
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
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsqrt_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 * 
 * Compute the square root of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
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
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsqrt_ru(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 * 
 * Compute the square root of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
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
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fsqrt_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
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
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 * 
 * Compute the reciprocal square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns
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
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __frsqrt_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-to-nearest-even mode.
 *
 * Divide two floating point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fdiv_rn(float x, float y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-towards-zero mode.
 *
 * Divide two floating point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fdiv_rz(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-up mode.
 * 
 * Divide two floating point values \p x by \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fdiv_ru(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-down mode.
 *
 * Divide two floating point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  __fdiv_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return the number of consecutive high-order zero bits in a 32 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 31) of \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of zero bits.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __clz(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 32 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1. 
 *
 * \return Returns a value between 0 and 32 inclusive representing the position of the first bit set.
 * - __ffs(0) returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __ffs(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 32 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of set bits.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __popc(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 32 bit unsigned integer.
 *
 * Reverses the bit order of the 32 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 31-N of \p x.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __brev(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of consecutive high-order zero bits in a 64 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 63) of \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of zero bits.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __clzll(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 64 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1. 
 *
 * \return Returns a value between 0 and 64 inclusive representing the position of the first bit set.
 * - __ffsll(0) returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __ffsll(long long int x);


/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 64 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of set bits.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __popcll(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 64 bit unsigned integer.
 *
 * Reverses the bit order of the 64 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 63-N of \p x.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __brevll(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return selected bytes from two 32 bit unsigned integers.
 *
 * byte_perm(x,y,s) returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two 
 * input integers \p x and \p y, as specified by a selector, \p s.
 *
 * The input bytes are indexed as follows:
 * <pre>
 * input[0] = x<7:0>   input[1] = x<15:8>
 * input[2] = x<23:16> input[3] = x<31:24>
 * input[4] = y<7:0>   input[5] = y<15:8>
 * input[6] = y<23:16> input[7] = y<31:24>
 * </pre>
 * The selector indices are as follows (the upper 16-bits of the selector are not used):
 * <pre>
 * selector[0] = s<2:0>  selector[1] = s<6:4>
 * selector[2] = s<10:8> selector[3] = s<14:12>
 * </pre>
 * \return The returned value r is computed to be:
 * <tt>result[n] := input[selector[n]]</tt>
 * where <tt>result[n]</tt> is the nth byte of r.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __byte_perm(unsigned int x, unsigned int y, unsigned int s);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute average of signed input arguments, avoiding overflow
 * in the intermediate sum.
 *
 * Compute average of signed input arguments \p x and \p y 
 * as ( \p x + \p y ) >> 1, avoiding overflow in the intermediate sum.
 *
 * \return Returns a signed integer value representing the signed 
 * average value of the two inputs.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __hadd(int, int);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute rounded average of signed input arguments, avoiding
 * overflow in the intermediate sum.
 *
 * Compute average of signed input arguments \p x and \p y 
 * as ( \p x + \p y + 1 ) >> 1, avoiding overflow in the intermediate
 * sum.
 *
 * \return Returns a signed integer value representing the signed 
 * rounded average value of the two inputs.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __rhadd(int, int);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute average of unsigned input arguments, avoiding overflow
 * in the intermediate sum.
 *
 * Compute average of unsigned input arguments \p x and \p y 
 * as ( \p x + \p y ) >> 1, avoiding overflow in the intermediate sum.
 *
 * \return Returns an unsigned integer value representing the unsigned 
 * average value of the two inputs.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __uhadd(unsigned int, unsigned int);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute rounded average of unsigned input arguments, avoiding
 * overflow in the intermediate sum.
 *
 * Compute average of unsigned input arguments \p x and \p y 
 * as ( \p x + \p y + 1 ) >> 1, avoiding overflow in the intermediate
 * sum.
 *
 * \return Returns an unsigned integer value representing the unsigned 
 * rounded average value of the two inputs.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __urhadd(unsigned int, unsigned int);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __double2int_rz(double);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __double2uint_rz(double);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          __double2ll_rz(double);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int __double2ull_rz(double);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __pm0(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __pm1(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __pm2(void);
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           __pm3(void);

/*******************************************************************************
 *                                                                             *
 *                                SIMD functions                               *
 *                                                                             *
 *******************************************************************************/

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword absolute value.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes,
 * then computes absolute value for each of parts.
 * Result is stored as unsigned int and returned. 
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabs2(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword absolute value with signed saturation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes,
 * then computes absolute value with signed saturation for each of parts.
 * Result is stored as unsigned int and returned. 
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsss2(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed addition, with wrap-around: a + b
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs unsigned addition on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vadd2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword addition with signed saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs addition with signed saturation on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vaddss2 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword addition with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs addition with unsigned saturation on corresponding parts.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vaddus2 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed rounded average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes signed rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vavgs2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned rounded average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes unsigned rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vavgu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes unsigned avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vhaddu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if they are equal, and 0000 otherwise.
 * For example __vcmpeq2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpeq2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a >= b ? 0xffff : 0.
 * 
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part >= 'b' part, and 0000 otherwise.
 * For example __vcmpges2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpges2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part >= 'b' part, and 0000 otherwise.
 * For example __vcmpgeu2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a > b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part > 'b' part, and 0000 otherwise.
 * For example __vcmpgts2(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xffff if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgts2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a > b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part > 'b' part, and 0000 otherwise.
 * For example __vcmpgtu2(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xffff if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a <= b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part <= 'b' part, and 0000 otherwise.
 * For example __vcmples2(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xffff if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmples2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0.
 *
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part <= 'b' part, and 0000 otherwise.
 * For example __vcmpleu2(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xffff if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpleu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a < b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part < 'b' part, and 0000 otherwise.
 * For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmplts2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a < b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part < 'b' part, and 0000 otherwise.
 * For example __vcmpltu2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpltu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part != 'b' part, and 0000 otherwise.
 * For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a != b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpne2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword absolute difference of unsigned integer computation: |a - b|
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute difference. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed maximum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes signed maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmaxs2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned maximum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes unsigned maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmaxu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes signed minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmins2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes unsigned minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vminu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part == 'b' part.
 * If both equalities are satisfiad, function returns 1.
 * \return Returns 1 if a = b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vseteq2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetges2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgeu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgts2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgtu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetles2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetleu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetlts2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetltu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part != 'b' part.
 * If both conditions are satisfied, function returns 1.
 * \return Returns 1 if a != b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetne2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword sum of abs diff of unsigned.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute differences, and returns
 * sum of those differences.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsadu2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed substraction, with wrap-around.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsub2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed substraction, with signed saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction with signed saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsubss2 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword substraction with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction with unsigned saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsubus2 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword negation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vneg2(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword negation with signed saturation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vnegss2(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword sum of absolute difference of signed integer.
 *
 * Splits 4 bytes of each into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute difference.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword sum of absolute difference of signed.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions computes absolute difference and sum it up. 
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsads2(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute value.
 *
 * Splits argument by bytes. Computes absolute value of each byte.
 * Result is stored as unsigned int.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabs4(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute value with signed saturation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte,
 * then computes absolute value with signed saturation for each of parts.
 * Result is stored as unsigned int and returned. 
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsss4(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed addition.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs unsigned addition on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vadd4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte addition with signed saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs addition with signed saturation on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vaddss4 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte addition with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs addition with unsigned saturation on corresponding parts.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vaddus4 (unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed rounder average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes signed rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vavgs4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned rounded average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes unsigned rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vavgu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes unsigned avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vhaddu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if they are equal, and 00 otherwise.
 * For example __vcmpeq4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a = b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpeq4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part >= 'b' part, and 00 otherwise.
 * For example __vcmpges4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpges4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part >= 'b' part, and 00 otherwise.
 * For example __vcmpgeu4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a = b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part > 'b' part, and 00 otherwise.
 * For example __vcmpgts4(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xff if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgts4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part > 'b' part, and 00 otherwise.
 * For example __vcmpgtu4(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xff if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part <= 'b' part, and 00 otherwise.
 * For example __vcmples4(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xff if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmples4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part <= 'b' part, and 00 otherwise.
 * For example __vcmpleu4(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xff if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpleu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part < 'b' part, and 00 otherwise.
 * For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmplts4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part < 'b' part, and 00 otherwise.
 * For example __vcmpltu4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpltu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part != 'b' part, and 00 otherwise.
 * For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a != b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vcmpne4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute difference of unsigned integer.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes absolute difference. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed maximum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes signed maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmaxs4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned maximum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes unsigned maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmaxu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed minimum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes signed minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vmins4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned minimum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes unsigned minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vminu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part == 'b' part.
 * If both equalities are satisfiad, function returns 1.
 * \return Returns 1 if a = b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vseteq4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetles4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 part, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetleu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetlts4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetltu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetges4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgeu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgts4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetgtu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part != 'b' part.
 * If both conditions are satisfied, function returns 1.
 * \return Returns 1 if a != b, else returns 0.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsetne4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte sum af abs difference of unsigned.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes absolute differences, and returns
 * sum of those differences.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsadu4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions performs substraction. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsub4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction with signed saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions performs substraction with signed saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsubss4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions performs substraction with unsigned saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsubus4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte negation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vneg4(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte negation with signed saturation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vnegss4(unsigned int a);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute difference of signed integer.
 *
 * Splits 4 bytes of each into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes absolute difference.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte sum of abs difference of signed.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions computes absolute difference and sum it up. 
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
__DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int __vsads4(unsigned int a, unsigned int b);

/*******************************************************************************
 *                                                                             *
 *                            END SIMD functions                               *
 *                                                                             *
 *******************************************************************************/
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ int mulhi(int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(unsigned int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long int mul64hi(long long int a, long long int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(long long int a, unsigned long long int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(unsigned long long int a, long long int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ int float_as_int(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float int_as_float(int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float saturate(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ int mul24(int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umul24(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ void trap(void);

/* argument is optional, value of 0 means no value */
__DEVICE_FUNCTIONS_STATIC_DECL__ void brkpt(int c = 0);

__DEVICE_FUNCTIONS_STATIC_DECL__ void syncthreads(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ void prof_trigger(int e);

__DEVICE_FUNCTIONS_STATIC_DECL__ void threadfence(bool global = true);

__DEVICE_FUNCTIONS_STATIC_DECL__ int float2int(float a, enum cudaRoundMode mode = cudaRoundZero);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int float2uint(float a, enum cudaRoundMode mode = cudaRoundZero);

__DEVICE_FUNCTIONS_STATIC_DECL__ float int2float(int a, enum cudaRoundMode mode = cudaRoundNearest);

__DEVICE_FUNCTIONS_STATIC_DECL__ float uint2float(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest);

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
__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_count(int predicate);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_and(int predicate);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __syncthreads_or(int predicate);

/*******************************************************************************
*                                                                              *
* MEMORY FENCE FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence_block();

__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence();

__DEVICE_FUNCTIONS_STATIC_DECL__ void __threadfence_system();

/*******************************************************************************
*                                                                              *
* VOTE FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int __all(int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __any(int a);

__DEVICE_FUNCTIONS_STATIC_DECL__
#if defined(__CUDACC_RTC__)
unsigned int
#else /* !__CUDACC_RTC__ */
int
#endif /* __CUDACC_RTC__ */
__ballot(int a);

/*******************************************************************************
*                                                                              *
* MISCELLANEOUS FUNCTIONS                                                      *
*                                                                              *
*******************************************************************************/
#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ void __brkpt(int);
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ void __brkpt();
#endif /* __CUDACC_RTC__ */

__DEVICE_FUNCTIONS_STATIC_DECL__
#if defined(__CUDACC_RTC__)
clock_t
#else /* !__CUDACC_RTC__ */
int
#endif /* __CUDACC_RTC__ */
clock();

__DEVICE_FUNCTIONS_STATIC_DECL__ long long clock64();
    
#define __prof_trigger(X) asm __volatile__ ("pmevent \t" #X ";")

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm0(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm1(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm2(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __pm3(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ void __trap(void);

__DEVICE_FUNCTIONS_STATIC_DECL__ void* memcpy(void *dest, const void *src, size_t n);

__DEVICE_FUNCTIONS_STATIC_DECL__ void* memset(void *dest, int c, size_t n);

/*******************************************************************************
*                                                                              *
* MATH FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int __clz(int x);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __clzll(long long x);

#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popc(unsigned int x);
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popc(int x);
#endif /* __CUDACC_RTC__ */

#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popcll(unsigned long long x);
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __popcll(long long x);
#endif /* __CUDACC_RTC__ */

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __byte_perm(unsigned int a,
                                                unsigned int b,
                                                unsigned int c);

/*******************************************************************************
*                                                                              *
* INTEGER MATH FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ int min(int x, int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umin(unsigned int x, unsigned int y);
    
__DEVICE_FUNCTIONS_STATIC_DECL__ long long llmin(long long x, long long y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long ullmin(unsigned long long x,
                                                 unsigned long long y);
    
__DEVICE_FUNCTIONS_STATIC_DECL__ int max(int x, int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umax(unsigned int x, unsigned int y);
    
__DEVICE_FUNCTIONS_STATIC_DECL__ long long llmax(long long x, long long y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long ullmax(unsigned long long x,
                                                 unsigned long long y);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __mulhi(int x, int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __umulhi(unsigned int x, unsigned int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __mul64hi(long long x, long long y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __umul64hi(unsigned long long x,
                                                     unsigned long long y);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __mul24(int x, int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __umul24(unsigned int x, unsigned int y);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __brev(unsigned int x);
    
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __brevll(unsigned long long x);
    
#if defined(__CUDACC_RTC__)
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __sad(int x, int y, unsigned int z);
#else /* !__CUDACC_RTC__ */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __sad(int x, int y, int z);
#endif /* __CUDACC_RTC__ */

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __usad(unsigned int x,
                                           unsigned int y,
                                           unsigned int z);

__DEVICE_FUNCTIONS_STATIC_DECL__ int abs(int x);

__DEVICE_FUNCTIONS_STATIC_DECL__ long labs(long x);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long llabs(long long x);

/*******************************************************************************
*                                                                              *
* FP MATH FUNCTIONS                                                            *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float floorf(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ double floor(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ float fabsf(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ double fabs(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __rcp64h(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float fmaxf(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float rsqrtf(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double fmin(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double fmax(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double rsqrt(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double ceil(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double trunc(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float exp2f(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float truncf(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float ceilf(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __saturatef(float x);

/*******************************************************************************
*                                                                              *
* FMAF                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rn(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rz(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_rd(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ru(float x, float y, float z);

/*******************************************************************************
*                                                                              *
* FMAF_IEEE                                                                    *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rn(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rz(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_rd(float x, float y, float z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmaf_ieee_ru(float x, float y, float z);

/*******************************************************************************
*                                                                              *
* FMA                                                                          *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rn(double x, double y, double z);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rz(double x, double y, double z);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_rd(double x, double y, double z);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __fma_ru(double x, double y, double z);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdividef(float x, float y);

/*******************************************************************************
*                                                                              *
* FDIV                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rn(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rz(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_rd(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fdiv_ru(float x, float y);

/*******************************************************************************
*                                                                              *
* FRCP                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rn(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rz(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_rd(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frcp_ru(float x);

/*******************************************************************************
*                                                                              *
* FSQRT                                                                        *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rn(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rz(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_rd(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsqrt_ru(float x);

/*******************************************************************************
*                                                                              *
* DDIV                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rn(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rz(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_rd(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ddiv_ru(double x, double y);

/*******************************************************************************
*                                                                              *
* DRCP                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rn(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rz(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_rd(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __drcp_ru(double x);

/*******************************************************************************
*                                                                              *
* DSQRT                                                                        *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rn(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rz(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_rd(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dsqrt_ru(double x);

__DEVICE_FUNCTIONS_STATIC_DECL__ float sqrtf(float x);

__DEVICE_FUNCTIONS_STATIC_DECL__ double sqrt(double x);

/*******************************************************************************
*                                                                              *
* DADD                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rn(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rz(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_rd(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dadd_ru(double x, double y);

/*******************************************************************************
*                                                                              *
* DMUL                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rn(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rz(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_rd(double x, double y);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __dmul_ru(double x, double y);

/*******************************************************************************
*                                                                              *
* FADD                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rd(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_ru(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rn(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fadd_rz(float x, float y);

/*******************************************************************************
*                                                                              *
* FMUL                                                                         *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rd(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_ru(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rn(float x, float y);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fmul_rz(float x, float y);

/*******************************************************************************
*                                                                              *
* CONVERSION FUNCTIONS                                                         *
*                                                                              *
*******************************************************************************/
/* double to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rn(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rz(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_rd(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __double2float_ru(double d);
    
/* double to int */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rn(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rz(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_rd(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2int_ru(double d);

/* double to uint */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rn(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rz(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_rd(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __double2uint_ru(double d);

/* int to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __int2double_rn(int i);

/* uint to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __uint2double_rn(unsigned int i);

/* float to int */
__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rn(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rz(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_rd(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float2int_ru(float in);

/* float to uint */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rn(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rz(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_rd(float in);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __float2uint_ru(float in);

/* int to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rn(int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rz(int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_rd(int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int2float_ru(int in);

/* unsigned int to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rn(unsigned int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rz(unsigned int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_rd(unsigned int in);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __uint2float_ru(unsigned int in);

/* hiloint vs double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __hiloint2double(int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2loint(double d);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __double2hiint(double d);

/* float to long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rn(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rz(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_rd(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __float2ll_ru(float f);

/* float to unsigned long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rn(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rz(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_rd(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __float2ull_ru(float f);

/* double to long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rn(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rz(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_rd(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long __double2ll_ru(double f);

/* double to unsigned long long */
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rn(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rz(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_rd(double f);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long __double2ull_ru(double f);

/* long long to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rn(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rz(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_rd(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ll2float_ru(long long l);

/* unsigned long long to float */
__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rn(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rz(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_rd(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __ull2float_ru(unsigned long long l);

/* long long to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rn(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rz(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_rd(long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ll2double_ru(long long l);

/* unsigned long long to double */
__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rn(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rz(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_rd(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ double __ull2double_ru(unsigned long long l);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned short __float2half_rn(float f);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __half2float(unsigned short h);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __int_as_float(int x);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __float_as_int(float x);
    
__DEVICE_FUNCTIONS_STATIC_DECL__ double __longlong_as_double(long long x);

__DEVICE_FUNCTIONS_STATIC_DECL__ long long  __double_as_longlong (double x);

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ float __sinf(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __cosf(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __log2f(float a);

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ float __tanf(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ void __sincosf(float a, float *sptr, float *cptr);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __expf(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __exp10f(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __log10f(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __logf(float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __powf(float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float fdividef(float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ double fdivide(double a, double b);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __hadd(int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __rhadd(int a, int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __uhadd(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __urhadd(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rn (float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rz (float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_rd (float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __fsub_ru (float a, float b);

__DEVICE_FUNCTIONS_STATIC_DECL__ float __frsqrt_rn (float a);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __ffs(int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ int __ffsll(long long int a);

/*******************************************************************************
*                                                                              *
* ATOMIC OPERATIONS                                                            *
*                                                                              *
*******************************************************************************/
__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicAdd(int *p, int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicAdd(unsigned int *p, unsigned int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicAdd(unsigned long long *p, unsigned long long val);

__DEVICE_FUNCTIONS_STATIC_DECL__
float __fAtomicAdd(float *p, float val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 600
__DEVICE_FUNCTIONS_STATIC_DECL__
double __dAtomicAdd(double *p, double val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 600 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicExch(int *p, int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicExch(unsigned int *p, unsigned int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicExch(unsigned long long *p,
                                   unsigned long long val);

__DEVICE_FUNCTIONS_STATIC_DECL__
float __fAtomicExch(float *p, float val);

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicMin(int *p, int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long __illAtomicMin(long long *p, long long val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicMin(unsigned int *p, unsigned int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicMin(unsigned long long *p, unsigned long long val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicMax(int *p, int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long __illAtomicMax(long long *p, long long val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicMax(unsigned int *p, unsigned int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long __ullAtomicMax(unsigned long long *p, unsigned long long val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicInc(unsigned int *p, unsigned int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicDec(unsigned int *p, unsigned int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicCAS(int *p, int compare, int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicCAS(unsigned int *p, unsigned int compare,
                          unsigned int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicCAS(unsigned long long int *p,
                                      unsigned long long int compare,
                                      unsigned long long int val);

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicAnd(int *p, int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicAnd(long long int *p, long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicAnd(unsigned int *p, unsigned int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicAnd(unsigned long long int *p,
                                      unsigned long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicOr(int *p, int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicOr(long long int *p, long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicOr(unsigned int *p, unsigned int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicOr(unsigned long long int *p,
                                     unsigned long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
int __iAtomicXor(int *p, int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
long long int __llAtomicXor(long long int *p, long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned int __uAtomicXor(unsigned int *p, unsigned int val);

#if !defined(__CUDACC_RTC__) || __CUDA_ARCH__ >= 320
__DEVICE_FUNCTIONS_STATIC_DECL__
unsigned long long int __ullAtomicXor(unsigned long long int *p,
                                      unsigned long long int val);
#endif /* !__CUDACC_RTC__ || __CUDA_ARCH__ >= 320 */

/*******************************************************************************
 *                                                                             *
 *                          SIMD functions                                     *
 *                                                                             *
 *******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabs2(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsss2(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vadd2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddss2 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddus2 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgs2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vhaddu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpeq2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpges2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgts2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmples2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpleu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmplts2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpltu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpne2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxs2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmins2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vminu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vseteq2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetges2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgeu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgts2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgtu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetles2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetleu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetlts2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetltu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetne2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsadu2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsub2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubss2 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubus2 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vneg2(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vnegss2(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsads2(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabs4(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsss4(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vadd4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddss4 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vaddus4 (unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgs4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vavgu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vhaddu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpeq4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpges4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgts4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmples4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpleu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmplts4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpltu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vcmpne4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxs4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmaxu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vmins4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vminu4(unsigned int a, unsigned int b);
__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vseteq4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetles4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetleu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetlts4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetltu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetges4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgeu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgts4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetgtu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsetne4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsadu4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsub4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubss4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsubus4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vneg4(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vnegss4(unsigned int a);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b);

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int __vsads4(unsigned int a, unsigned int b);

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

#if !defined(__CUDACC_RTC__)
#include "device_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#include "device_atomic_functions.h"
#include "device_double_functions.h"
#include "sm_20_atomic_functions.h"
#include "sm_32_atomic_functions.h"
#include "sm_35_atomic_functions.h"
#include "sm_20_intrinsics.h"
#include "sm_30_intrinsics.h"
#include "sm_32_intrinsics.h"
#include "sm_35_intrinsics.h"
#include "surface_functions.h"
#include "texture_fetch_functions.h"
#include "texture_indirect_functions.h"
#include "surface_indirect_functions.h"

#endif /* !__DEVICE_FUNCTIONS_H__ */
