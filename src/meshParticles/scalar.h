#ifndef SCALAR_H
#define SCALAR_H

#include "libbell.h"

#include <float.h>
#include <assert.h>
#include <cmath>

typedef float scalar;

#define SCALAR_MAX     FLT_MAX
#define SCALAR_MIN     FLT_MIN
#define SCALAR_EPSILON FLT_EPSILON
#define FABS(x)        fabsf(x)

#define SQRT(x)        sqrtf(x)

#define COS(x)         cosf(x)
#define SIN(x)         sinf(x)
#define TAN(x)         tanf(x)

#define ACOS(x)         acosf(x)
#define ASIN(x)         asinf(x)
#define ATAN(x)         atanf(x)

#define SIGN(x)         ((x < 0) ? (scalar)-1.0 : ((x > 0) ? (scalar) 1.0 : (scalar) 0.0))

#define rand_scalar(x)  (x * rand() / (scalar) RAND_MAX)
#define rand_int(x)     (int) ((scalar) (x+1) * (rand() / (scalar) (RAND_MAX + 1.0)))

/*
 * Performing floating point to integer conversion is *very* slow on x86 systems.
 * Here FLOOR and CEIL are implemented using lrint on linux platforms and in asm on windows.
 *
 * Code and techniques are from:
 *
 *      "Fast Rounding of Floating Point Numbers in C/C++ on Wintel Platform"
 *       Laurent de Soras 2004
 *       http://ldesoras.free.fr
 *
 */

#ifdef _WIN64 
__inline long int lrint(double x)
{
	return (long int) (x);
}
__inline long int lrintf(float x)
{
	return (long int) (x);
}
#elif defined(WIN32)
__inline long int lrintf(float x){
	int i; 
	__asm { fld x 
	fistp i 
	} 
	return (i); 
}

__inline long int lrint(double x){
	int i; 
	__asm { fld x 
	fistp i 
	} 
	return (i); 
}
#endif

inline int ROUND(float x){ return lrintf(x); }
inline int FLOOR(float x){ return  (lrintf(2*x-0.5F) >> 1); }
inline int CEIL(float x) { return -(lrintf(-0.5F-2*x) >> 1); }

inline int ROUND(double x){ return lrint(x); }
inline int FLOOR(double x){ return  (lrint(2*x-0.5) >> 1); }
inline int CEIL(double x) { return -(lrint(-0.5-2*x) >> 1); }


#ifndef M_PI //for WIN32
#define M_PI 3.14159265358979323846
#endif

#endif
