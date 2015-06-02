#ifndef GLOBAL_H
#define GLOBAL_H

#define MAX(a, b) ((a)>=(b) ? (a) : (b))
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

typedef unsigned int uint;

#define drand48() (rand() / RAND_MAX)

//#define USE_ASSERT
//#define DEBUG_MODE

//#define HIDE_FROZEN_PARTICLES
//#define USE_VBO_CPU
#define USE_CUDA
#define USE_VBO_CUDA
//#define USE_FFMPEG
//#define DIRECT_FORCING

enum Dimensions
{
	GRID_RESOLUTION = 1024		  // < 16-bit grid, must be power-of-two
};

enum SpecificSimAlgorithm
{
	SPH_BASIC = 0,
	SPH_PENALTY_FORCE,
	SPH_DIRECT_FORCE,
	SPH_IHMSEN,
	SPH_APPROXIMATE,
	WCSPH_BASIC,
	WCSPH_PENALTY_FORCE,
	WCSPH_DIRECT_FORCE,
	WCSPH_IHMSEN,
	WCSPH_VERSATILECOUPLING,
	WCSPH_SPH_APPROXIMATE,
	PCISPH_BASIC,
	PCISPH_PENALTY_FORCE,
	PCISPH_DIRECT_FORCE,
	PCISPH_IHMSEN,
	PCISPH_VERSATILECOUPLING
};

const unsigned int max_neighs = 50; // Usually each particle has 30-40 neighs 
const unsigned int MAX_PCISPH_LOOPS	= 50; // 50 //3 TODO: need to determine this value for performance/visual effects
const unsigned int MIN_PCISPH_LOOPS = 3;

class UnifiedConstants;
class UnifiedPhysics;

extern UnifiedConstants *fc;
extern UnifiedPhysics *myFluid;

#endif
