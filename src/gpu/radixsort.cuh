/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Radixsort project which demonstrates the use of CUDA in a multi phase
 * sorting computation.
 * Type definitions.
 */

#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#include "host_defines.h"
#include "cutil.h"
#include "vector_types.h"
#include <limits>

#define SYNCIT __syncthreads()

// Use 16 bit keys/values
#define SIXTEEN 0

typedef unsigned int uint;
typedef unsigned short ushort;

#if SIXTEEN
typedef struct __align__(4) {
    ushort key;
    ushort value;
#else
typedef struct __align__(8) {
    uint key;
    uint value;
#endif
} KeyValuePair;




/*======================================================================================================*/
/*	DEFINITIONS ADDED FROM radixsort_kernel.cu */
/*======================================================================================================*/


static const int NUM_SMS = 16;
static const int NUM_THREADS_PER_SM = 192;
static const int NUM_THREADS_PER_BLOCK = 64;
//static const int NUM_THREADS = NUM_THREADS_PER_SM * NUM_SMS;
static const int NUM_BLOCKS = (NUM_THREADS_PER_SM / NUM_THREADS_PER_BLOCK) * NUM_SMS;
static const int RADIX = 8;                                                        // Number of bits per radix sort pass
static const int RADICES = 1 << RADIX;                                             // Number of radices
static const int RADIXMASK = RADICES - 1;                                          // Mask for each radix sort pass
#if SIXTEEN
static const int RADIXBITS = 16;                                                   // Number of bits to sort over
#else
static const int RADIXBITS = 32;                                                   // Number of bits to sort over
#endif
static const int RADIXTHREADS = 16;                                                // Number of threads sharing each radix counter
static const int RADIXGROUPS = NUM_THREADS_PER_BLOCK / RADIXTHREADS;               // Number of radix groups per CTA
static const int TOTALRADIXGROUPS = NUM_BLOCKS * RADIXGROUPS;                      // Number of radix groups for each radix
static const int SORTRADIXGROUPS = TOTALRADIXGROUPS * RADICES;                     // Total radix count
static const int GRFELEMENTS = (NUM_THREADS_PER_BLOCK / RADIXTHREADS) * RADICES; 
static const int GRFSIZE = GRFELEMENTS * sizeof(uint); 

// Prefix sum variables
static const int PREFIX_NUM_THREADS_PER_SM = NUM_THREADS_PER_SM;
static const int PREFIX_NUM_THREADS_PER_BLOCK = PREFIX_NUM_THREADS_PER_SM;
static const int PREFIX_NUM_BLOCKS = (PREFIX_NUM_THREADS_PER_SM / PREFIX_NUM_THREADS_PER_BLOCK) * NUM_SMS;
static const int PREFIX_BLOCKSIZE = SORTRADIXGROUPS / PREFIX_NUM_BLOCKS;
static const int PREFIX_GRFELEMENTS = PREFIX_BLOCKSIZE + 2 * PREFIX_NUM_THREADS_PER_BLOCK;
static const int PREFIX_GRFSIZE = PREFIX_GRFELEMENTS * sizeof(uint);

// Shuffle variables
static const int SHUFFLE_GRFOFFSET = RADIXGROUPS * RADICES;
static const int SHUFFLE_GRFELEMENTS = SHUFFLE_GRFOFFSET + PREFIX_NUM_BLOCKS; 
static const int SHUFFLE_GRFSIZE = SHUFFLE_GRFELEMENTS * sizeof(uint); 


__global__ 
void RadixSum(KeyValuePair *pData, uint elements, uint elements_rounded_to_3072, uint shift);

__global__ 
void RadixPrefixSum();

__global__ 
void RadixAddOffsetsAndShuffle(KeyValuePair* pSrc, KeyValuePair* pDst, uint elements, uint elements_rounded_to_3072, int shift);



extern "C" {
    void RadixSort(KeyValuePair *pData0, KeyValuePair *pData1, uint elements, uint bits);
	}

#endif // #ifndef _RADIXSORT_H_
