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

#if !defined(__SM_20_INTRINSICS_H__)
#define __SM_20_INTRINSICS_H__

#if defined(__CUDACC_RTC__)
#define __SM_20_INTRINSICS_DECL__ __host__ __device__
#else /* __CUDACC_RTC__ */
#define __SM_20_INTRINSICS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

#if !defined(__CUDACC_RTC__)
extern "C"
{
#endif /* !__CUDACC_RTC__ */
extern __device__ __device_builtin__ void                   __threadfence_system(void);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-to-nearest-even mode.
 *
 * Divides two floating point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-towards-zero mode.
 *
 * Divides two floating point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-up mode.
 * 
 * Divides two floating point values \p x by \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-down mode.
 *
 * Divides two floating point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rd(double x);
extern __device__ __device_builtin__ unsigned int          __ballot(int);
extern __device__ __device_builtin__ int                   __syncthreads_count(int);
extern __device__ __device_builtin__ int                   __syncthreads_and(int);
extern __device__ __device_builtin__ int                   __syncthreads_or(int);
extern __device__ __device_builtin__ long long int         clock64(void);

/* The following intrinsics are special versions of __fmaf_r?() that always 
 * map to a single-precision FMA instruction in non-FTZ mode, regardless of
 * the compiler's -ftz flag setting. These are supported for NVIDIA internal 
 * library work only.
 */
extern __device__ __device_builtin__ float                 __fmaf_ieee_rn(float, float, float);
extern __device__ __device_builtin__ float                 __fmaf_ieee_rz(float, float, float);
extern __device__ __device_builtin__ float                 __fmaf_ieee_ru(float, float, float);
extern __device__ __device_builtin__ float                 __fmaf_ieee_rd(float, float, float);
extern __device__ __device_builtin__ double                __rcp64h(double);

#if !defined(__CUDACC_RTC__)
}
#endif /* !__CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SM_20_INTRINSICS_DECL__ unsigned int ballot(bool pred);

__SM_20_INTRINSICS_DECL__ int syncthreads_count(bool pred);

__SM_20_INTRINSICS_DECL__ bool syncthreads_and(bool pred);

__SM_20_INTRINSICS_DECL__ bool syncthreads_or(bool pred);


// This function returns 1 if generic address "ptr" is in global memory space.
// It returns 0 if "ptr" is in shared, local or constant memory space.
__SM_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr);

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_20_INTRINSICS_DECL__

#if !defined(__CUDACC_RTC__)
#include "sm_20_intrinsics.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__SM_20_INTRINSICS_H__ */
