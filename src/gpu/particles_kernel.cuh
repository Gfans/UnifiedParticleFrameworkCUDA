#ifndef GPU_UNIFIED_PARTICLES_KERNEL_H_
#define GPU_UNIFIED_PARTICLES_KERNEL_H_

#include "particles.cuh"
#include "radixsort.cuh"
#include "sm_35_atomic_functions.h"
#include "vector_functions.h"

//=====================================================================
//                      CUDA TEXTURES & ARRAYS DECLARATION 
//=====================================================================
extern texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_pos_zindex;
extern texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_relative_pos;
extern texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_predicted_pos;
extern texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_vel;
extern texture<float4,          cudaTextureType1D,  cudaReadModeElementType> texture_static_force;
extern texture<float4,          cudaTextureType1D,  cudaReadModeElementType> texture_dynamic_boundary_force;
extern texture<float4,          cudaTextureType1D,  cudaReadModeElementType> texture_corr_pressure_force;
extern texture<int,				cudaTextureType1D,	cudaReadModeElementType> texture_type;
extern texture<int,				cudaTextureType1D,	cudaReadModeElementType> texture_active_type;
extern texture<int,				cudaTextureType1D,	cudaReadModeElementType> texture_parent_rb;
extern texture<int,				cudaTextureType1D,	cudaReadModeElementType> texture_order_in_child_particles_array;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_density;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_pressure;
extern texture<float,			cudaTextureType1D,  cudaReadModeElementType> texture_weighted_volume;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_corr_pressure;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_previous_corr_pressure;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_density_error;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_predicted_density;
extern texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_previous_predicted_density;
extern texture<unsigned int,	cudaTextureType2D,	cudaReadModeElementType> texture_zindex_array;

extern cudaArray* array_zindex;

//=====================================================================
//                      GRID SIZES & C++ VARIABLES DECLARATION
//=====================================================================
extern int num_blocks;
extern int gridsize_indices;

extern __constant__ float deltaT;

#if defined(SPH_PROFILING) || defined(SPH_PROFILING_VERBOSE)

	extern unsigned int frameCounter_;
	extern float        simulationTimeCounter_;
	extern float        surfaceExtractionTimeCounter_;
	extern float		parallelSortingTimeCounter_;
	extern float		blockGenerationTimeCounter_;
	extern float		externalForceCalculationTimeCounter_;
	extern float		pciLoopTimeCounter_;
	extern float		timeIntegrationTimeCounter_;
	extern float		indexCalculationTimeCounter_;
#endif

//=====================================================================
//                      DEVICE VARIABLES
//=====================================================================

extern __device__ unsigned int filteredCount;
extern __device__ unsigned int blockCount;

//=====================================================================
//          CONSTANT VARIABLES
//=====================================================================

extern __constant__ unsigned int xMASK;
extern __constant__ unsigned int yMASK;
extern __constant__ unsigned int zMASK;


extern __constant__ unsigned int PCOUNT;			// number of particles in simulation
extern __constant__ unsigned int PCOUNTROUNDED;		// number of particles in simulation rounded
extern __constant__ unsigned int RIGIDBODYCOUNT;	// number of rigid bodies in simulation

extern __constant__ float globalSupportRadius;
extern __constant__ float invglobalSupportRadius;
extern __constant__ float globalSupportRadius2;
extern __constant__ float d_const_particle_radius;
extern __constant__ float d_const_dist_to_center_mass_cutoff;
extern __constant__ float d_const_vel_cutoff;
extern __constant__ float d_const_terminal_speed;
extern __constant__ float fluidRestDensity;
extern __constant__ float invfluidRestDensity;
extern __constant__ float fluidRestVolume;
extern __constant__ float gamma;
extern __constant__ float fluidGasConstant;
extern __constant__ float fluidGasConstantWCSPH;
extern __constant__ float fluidViscosityConstant;
extern __constant__ float fluidViscosityConstant_tube;
extern __constant__ float gravityConstant;
extern __constant__ float3 scales;
extern __constant__ float3 zindexStartingVec;
extern __constant__ unsigned int gridResolution;
extern __constant__ unsigned int block_size;
extern __constant__ float invblock_size;
extern __constant__ unsigned int block_size3;
extern __constant__ unsigned int lutSize;
extern __constant__ float invlutSize;
//extern __constant__ float globalSupportRadiusByLutSize;
extern __constant__ float kernelSelf;

// PCISPH
extern __constant__ float densityErrorFactor;

extern __constant__ float initialMass;
extern __constant__ float initialMass2;
extern __constant__ float invinitialMass;
extern __constant__ float initialMassIntoGravityConstant;

extern __constant__ int intMax;

extern __constant__ bool addBoundaryForce;
// Collision Box
extern __constant__ float3 minCollisionBox;
extern __constant__ float3 maxCollisionBox;
// Virtual Z-index Bounding Box
extern __constant__ float3 minBoundingBox;
extern __constant__ float3 maxBoundingBox;
// Real Container Box
extern __constant__ float3 d_const_min_container_box;
extern __constant__ float3 d_const_max_container_box;

// Wall Weight Function Method
extern __constant__ float d_const_box_length;
extern __constant__ float d_const_box_height;
extern __constant__ float d_const_box_width;

extern __constant__ float wallX;

extern __constant__ unsigned int dimX;
extern __constant__ unsigned int dimY;
extern __constant__ unsigned int dimZ;
extern __constant__ unsigned int total_blocks;
extern __constant__ unsigned int maxArrayLength;

extern __constant__ float forceDistance;
extern __constant__ float invforceDistance;
extern __constant__ float deltaT;
extern __constant__ float deltaTWCSPH;
extern __constant__ float maxBoundaryForce;
extern __constant__ float spacing;
extern __constant__ float lutKernelM4Table[LUTSIZE];
extern __constant__ float lutKernelPressureGradientTable[LUTSIZE];
extern __constant__ float lutKernelViscosityLapTable[LUTSIZE];
extern __constant__ float lutSplineSurfaceTensionTable[LUTSIZE];
extern __constant__ float lutSplineSurfaceAdhesionTable[LUTSIZE];


extern __constant__ float CENTER_OF_MASS_THRESHOLD;
extern __constant__ unsigned int N_NEIGHBORS_THRESHOLD;

// rigid body
extern __constant__ float d_const_spring_coefficient;
extern __constant__ float d_const_spring_boundary_coefficient;
extern __constant__ float d_const_damping_coefficient;

// surface tension & Adhesion
extern __constant__ float d_const_surface_tension_gamma;
extern __constant__ float d_const_surface_adhesion_beta;

// Pipe points for second demo set up
extern __constant__ float3 pipePoint1;
extern __constant__ float3 pipePoint2;
extern __constant__ float pipeLength;
extern __constant__ float invpipeLength;
extern __constant__ float pipeLength2;
extern __constant__ float pipeRadius;

extern __constant__ float particleRenderingSize; 

// mod without divide, works on values from 0 upto 2m
#define WRAP(x,m) (((x)<m) ? (x) : (x-m))


__global__ void GetReductionFinalMax(float* idata, int numPnts, float* max_predicted_density);

//========================================================================
//								DEVICE FUNCTIONS
//========================================================================


// In all device functions, textures are assumed as to which one to use to 
// avoid passing them as arguments

template< typename T1, typename T2 >
__device__
	inline T1 operator + (const T1 &a, const T2 &b)
{
	T1 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
}

template< typename T1, typename T2 >
__device__
	inline T1 operator - (const T1 &a, const T2 &b)
{
	T1 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	return r;
}

template< typename T1, typename T2 >
__device__
	inline void operator -= (T1 &a, const T2 &b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

template< typename T1, typename T2 >
__device__
	inline void operator += (T1 &a, const T2 &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

template< typename T >
__device__
	inline void operator *= (T &a, const float &b )
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

template< typename T >
__device__
	inline T operator * (const T& a, const float& b )
{
	T r;
	r.x = a.x * b;
	r.y = a.y * b;
	r.z = a.z * b;
	return r;
}

template< typename T >
__device__
	inline T cross( const T& a, const T& b )
{
	T r;
	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;
	return r;
}

template< typename T >
__device__
	inline float dot( const T& a, const T& b )
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

template< typename T1, typename T2 >
__device__
	inline float distanceSq( const T1& a, const T2& b )
{
	float x = a.x - b.x;
	float y = a.y - b.y;
	float z = a.z - b.z;
	return ( x * x + y * y + z * z );
}

template< typename T1, typename T2 >
__device__
	inline float distanceSqrt( const T1& a, const T2& b )
{
	float x = a.x - b.x;
	float y = a.y - b.y;
	float z = a.z - b.z;
	return sqrt( x * x + y * y + z * z );
}

template< typename T1 >
__device__
	inline float distanceSq( const T1& a )
{
	return ( a.x * a.x + a.y * a.y + a.z * a.z );
}

template< typename T1 >
__device__
	inline float distanceSqrt( const T1& a )
{
	return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
}

template< typename T >
__device__
	inline void normalize( T& a )
{
	float magnitude = sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
	a.x /= magnitude;
	a.y /= magnitude;
	a.z /= magnitude;
}

__device__
	inline float4 CalculateBoundaryForcePerLiquidParticleDevice(const float4& position)
{
	float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef SPH_DEMO_SCENE_2

	if( position.x < wallX + forceDistance && position.y < 0.3  )
	{
		f += (make_float4(1.0,0.0,0.0,0.0) * ((wallX + forceDistance - position.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

#else

	if( position.x < minCollisionBox.x + forceDistance )
	{
		f += (make_float4(1.0,0.0,0.0,0.0) * ((minCollisionBox.x + forceDistance - position.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

#endif

	if( position.x > maxCollisionBox.x - forceDistance )
	{
		f += (make_float4(-1.0,0.0,0.0,0.0) * ((position.x + forceDistance - maxCollisionBox.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.y < minCollisionBox.y + forceDistance )
	{
		f += (make_float4(0.0,1.0,0.0,0.0) * ((minCollisionBox.y + forceDistance - position.y) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.y > maxCollisionBox.y - forceDistance )
	{
		f += (make_float4(0.0,-1.0,0.0,0.0) * ((position.y + forceDistance - maxCollisionBox.y) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.z < minCollisionBox.z + forceDistance )
	{
		f += (make_float4(0.0,0.0,1.0,0.0) * ((minCollisionBox.z + forceDistance - position.z) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.z > maxCollisionBox.z - forceDistance )
	{
		f += (make_float4(0.0,0.0,-1.0,0.0) * ((position.z + forceDistance - maxCollisionBox.z) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	return f;
}

__device__
	inline float4 CalculateBoundaryForcePerRigidParticleDevice(const float4& position, const float4& vel)
{
	float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	bool collisionOccured = false;

	// Left wall
	if( position.x - d_const_particle_radius < minCollisionBox.x )
	{
		collisionOccured = true;
		f += make_float4(1.0,0.0,0.0,0.0) * (minCollisionBox.x + d_const_particle_radius - position.x) * d_const_spring_boundary_coefficient;
	}

	// Right wall
	if( position.x + d_const_particle_radius > maxCollisionBox.x )
	{
		collisionOccured = true;
		f += make_float4(-1.0,0.0,0.0,0.0) * (position.x + d_const_particle_radius - maxCollisionBox.x) * d_const_spring_boundary_coefficient;
	}

	// Ground collision
	if( position.y - d_const_particle_radius < minCollisionBox.y )
	{
		collisionOccured = true;
		f += make_float4(0.0,1.0,0.0,0.0) * (minCollisionBox.y + d_const_particle_radius - position.y) * d_const_spring_boundary_coefficient;
	}

	// Ceil collision
	if( position.y + d_const_particle_radius > maxCollisionBox.y )
	{
		collisionOccured = true;
		f += make_float4(0.0,-1.0,0.0,0.0) * (position.y + d_const_particle_radius - maxCollisionBox.y) * d_const_spring_boundary_coefficient;
	}

	// Back wall
	if( position.z - d_const_particle_radius < minCollisionBox.z )
	{
		collisionOccured = true;
		f += make_float4(0.0,0.0,1.0,0.0) * (minCollisionBox.z + d_const_particle_radius - position.z) * d_const_spring_boundary_coefficient;
	}

	// Front wall
	if( position.z + d_const_particle_radius > maxCollisionBox.z )
	{
		collisionOccured = true;
		f += make_float4(0.0,0.0,-1.0,0.0) * (position.z + d_const_particle_radius - maxCollisionBox.z) * d_const_spring_boundary_coefficient;
	}

	if (collisionOccured)
	{
		f.x -= d_const_damping_coefficient * vel.x;
		f.y -= d_const_damping_coefficient * vel.y;
		f.z -= d_const_damping_coefficient * vel.z;
	}	

	return f;
}

__device__
	inline float4 AddBoundaryForcePCISPH(const float4& position)
{
	float4 force_dummy = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	if (position.x < minCollisionBox.x + forceDistance)
	{
		force_dummy = (make_float4(1.0,0.0,0.0,0.0) * ((minCollisionBox.x + forceDistance - position.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.x > maxCollisionBox.x - forceDistance )
	{
		force_dummy = (make_float4(-1.0,0.0,0.0,0.0) * ((position.x + forceDistance - maxCollisionBox.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}
	
	if( position.y < minCollisionBox.y + forceDistance )
	{
		force_dummy = (make_float4(0.0,1.0,0.0,0.0) * ((minCollisionBox.y + forceDistance - position.y) * invforceDistance * 2.0 * maxBoundaryForce));
	}
	if( position.y > maxCollisionBox.y - forceDistance )
	{
		force_dummy = (make_float4(0.0,-1.0,0.0,0.0) * ((position.y + forceDistance - maxCollisionBox.y) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.z < minCollisionBox.z + forceDistance )
	{
		force_dummy = (make_float4(0.0,0.0,1.0,0.0) * ((minCollisionBox.z + forceDistance - position.z) * invforceDistance * 2.0 * maxBoundaryForce));
	}
	if( position.z > maxCollisionBox.z - forceDistance )
	{
		force_dummy = (make_float4(0.0,0.0,-1.0,0.0) * ((position.z + forceDistance - maxCollisionBox.z) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	return force_dummy;
}


__device__ 
	inline void CollisionHandlingBox( float4& position, float4& velocity )
{

	float damping;
#ifdef SPH_DEMO_SCENE_2
	damping = -0.1;
#else
	damping = -0.5;
#endif

#ifdef SPH_DEMO_SCENE_2
	if(position.x < wallX && position.y < 0.3 )
	{
		position.x = wallX;
		velocity.x *= damping;
	}
#else
	if(position.x < minCollisionBox.x)
	{
		position.x = minCollisionBox.x;
		velocity.x *= damping;
	}
#endif
	if(position.x > maxCollisionBox.x)
	{
		position.x = maxCollisionBox.x;
		velocity.x *= damping;
	}
	if(position.y < minCollisionBox.y)
	{
		position.y = minCollisionBox.y;
		velocity.y *= damping;
	}
	if(position.y > maxCollisionBox.y)
	{
		position.y = maxCollisionBox.y;
		velocity.y *= damping;
	}
	if(position.z < minCollisionBox.z)
	{
		position.z = minCollisionBox.z;
		velocity.z *= damping;
	}
	if(position.z > maxCollisionBox.z)
	{
		position.z = maxCollisionBox.z;
		velocity.z *= damping;
	}	
}

__device__
	inline void BoundaryHandlingBoxPerNonRigidParticle_d(const float damping, float4& pos, float4& vel)
{
	// ground
	if (pos.y < minCollisionBox.y)
	{
		pos.y = minCollisionBox.y;
		vel.y *= damping;
	}
	
	// ceiling
	if (pos.y > maxCollisionBox.y)
	{
		pos.y = maxCollisionBox.y;
		vel.y *= damping;
	}

	// left wall
	if (pos.x < minCollisionBox.x)
	{
		pos.x = minCollisionBox.x;
		vel.x *= damping;
	}

	// right wall
	if (pos.x > maxCollisionBox.x)
	{
		pos.x = maxCollisionBox.x;
		vel.x *= damping;
	}
	
	// back wall
	if (pos.z < minCollisionBox.z)
	{
		pos.z = minCollisionBox.z;
		vel.z *= damping;
	}

	// front wall
	if (pos.z > maxCollisionBox.z)
	{
		pos.z = maxCollisionBox.z;
		vel.z *= damping;
	}
}

__device__
	inline bool IsParticleInPipe(float4& position)
{
	return ( position.x < pipePoint2.x );
}

__device__
	inline void CollisionHandlingCylinder( float4& position, float4& velocity )
{
	if( position.x < pipePoint2.x )
	{
		float3 axis = pipePoint2 - pipePoint1;
		float3 pos = make_float3(position.x, position.y, position.z);

		float val1 = distanceSqrt( cross( axis, pos - pipePoint1) );
		float d = val1 * invpipeLength;

		if( d > pipeRadius )
		{
			float t = -dot(axis, pipePoint1 - pos) * invpipeLength * invpipeLength;
			float3 pointOnLine = pipePoint1 + axis * t;

			float3 dir = pos - pointOnLine;
			normalize(dir);

			float3 newVal= pos - dir * (d - pipeRadius);
			position.x = newVal.x;
			position.y = newVal.y;
			position.z = newVal.z;

			newVal = dir * 2.0 * dot( dir, make_float3(velocity.x, velocity.y, velocity.z) );
			velocity.x -= newVal.x;
			velocity.y -= newVal.y;
			velocity.z -= newVal.z;
		}
	}
}

__device__
	inline unsigned int CalcIndex( const unsigned int& x, const unsigned int& y, const unsigned int& z)
{ 

	return (0 | tex2D( texture_zindex_array, 0, x ) | tex2D( texture_zindex_array, 1, y ) | tex2D( texture_zindex_array, 2, z ));
}

__device__ 
	inline unsigned int IncrementX( const unsigned int& i )
{
	return (((i | xMASK) + 1) & ~xMASK) | (i & xMASK); 
}

__device__ 
	inline unsigned int IncrementY( const unsigned int& i )
{
	return (((i | yMASK) + 1) & ~yMASK) | (i & yMASK); 
}

__device__ 
	inline unsigned int IncrementZ( const unsigned int& i )
{
	return (((i | zMASK) + 1) & ~zMASK) | (i & zMASK); 
}

// Wall Weight Function Method
__device__
	inline float DistanceToWallDevice(float4 pos)
{
	// To obtain the distance to the wall boundary, we have to compute the distance from each particle to 
	// all the polygons belonging to the wall boundary and select the minimum distance.
	float result = fmaxf(d_const_box_length, fmaxf(d_const_box_height, d_const_box_width) );
	float dist = 0.0f;
	const float x = pos.x;
	const float y = pos.y;
	const float z = pos.z;
	const float3 minContainerBox = d_const_min_container_box;
	const float3 maxContainerBox = d_const_max_container_box;
	// compare with dist to ground
	dist = fmaxf( (y - minContainerBox.y), 0.0f);
	result = fminf(result, dist);

	// compare with dist to ceil
	dist = fmaxf( (maxContainerBox.y - y), 0.0f);
	result = fminf(result, dist);

	// compare with dist to left wall
	dist = fmaxf( (x - minContainerBox.x), 0.0f);
	result = fminf(result, dist);

	// compare with dist to right wall
	dist = fmaxf( (maxContainerBox.x - x), 0.0f);
	result = fminf(result, dist);

	// compare with dist to back wall
	dist = fmaxf( (z - minContainerBox.z), 0.0f);
	result = fminf(result, dist);

	// compare with dist to front wall
	dist = fmaxf( (maxContainerBox.z - z), 0.0f);
	result = fminf(result, dist);

	return result;
}

__device__
	inline float WallWeightDevice(float distToWall)
{
	const float effectiveDist = globalSupportRadius - d_const_particle_radius;
	if (distToWall >= effectiveDist)
	{
		return 0.0f;
	}
	else
	{
		// first we determine distances to the maximum 10 potential boundary neighs particles' positions (actually 8 is enough)
		// see Figure 2 Wall weight functions from "Improvement in the Boundary Conditions of Smoothed Particle Hydrodynamics"
		float d1 = distToWall + d_const_particle_radius;
		float d2 = d1 + spacing;
		float d3 = sqrtf(d1 * d1 + spacing * spacing);		// symmetric 
		float d4 = sqrtf(d1 * d1 + spacing * spacing * 4);	// symmetric 
		float d5 = sqrtf(d2 * d2 + spacing * spacing);		// symmetric 
		float d6 = sqrtf(d2 * d2 + spacing * spacing * 4);	// symmetric 

		// TODO: use LUT in here or use direct computing ??? Which one is more efficient?
		uint tempValue = d1 * lutSize * invglobalSupportRadius;
		float kernelValue1 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		tempValue = d2 * lutSize * invglobalSupportRadius;
		float kernelValue2 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		tempValue = d3 * lutSize * invglobalSupportRadius;
		float kernelValue3 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		tempValue = d4 * lutSize * invglobalSupportRadius;
		float kernelValue4 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		tempValue = d5 * lutSize * invglobalSupportRadius;
		float kernelValue5 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		tempValue = d6 * lutSize * invglobalSupportRadius;
		float kernelValue6 = (tempValue >= lutSize) ? 0.0 : lutKernelM4Table[tempValue];

		float result = initialMass * ( kernelValue1 + kernelValue2 + 2 * ( kernelValue3 + kernelValue4 + kernelValue5 + kernelValue6) );
		return result;
	}
}

__device__
	inline float4 CalculateViscosityForce(uint dist_lut, float4 p_vel, float4 neigh_vel, float pVol, float neigh_invdensity)
{
	// compute artificial viscosity according to MCG03

	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef SPH_DEMO_SCENE_2

	float tmpVis = 0.0;
	if(p_pos_zindex.x < pipePoint2.x)
		tmpVis += fluidViscosityConstant_tube;
	else
		tmpVis += fluidViscosityConstant;

	if(neigh_pos_zindex[j].x < pipePoint2.x)
		tmpVis += fluidViscosityConstant_tube;
	else
		tmpVis += fluidViscosityConstant;

	tmpVis /= 2.0;
	//tmpVis = 50.0;

	force = -((p_vel - neigh_vel) * tmpVis * pVol * press_grad * initialMass * neigh_invdensity);

#else

	float kernel_visc_lap = lutKernelViscosityLapTable[dist_lut];
	force = ((p_vel - neigh_vel) * fluidViscosityConstant * pVol * kernel_visc_lap * initialMass * neigh_invdensity) * (-1.0f);		// Note: don't forget the force is negative

#endif

	return force;	

}

__device__
	inline float4 CalculateViscosityForcePCISPH(uint dist_lut, float4 p_vel, float4 neigh_vel, float pVol)
{
	// compute artificial viscosity according to MCG03

	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float kernel_visc_lap = lutKernelViscosityLapTable[dist_lut];
	force = ((p_vel - neigh_vel) * fluidViscosityConstant * pVol * pVol * kernel_visc_lap) * (-1.0f);		// Note: don't forget the force is negative

	return force;	
}

__device__
	inline float4 CalculateSurfaceTensionForcePCISPHDevice(const uint dist_lut, const float dist, const float4 p_pos, const float4 neigh_pos)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	const float splineCoefficient = lutSplineSurfaceTensionTable[dist_lut];

	force = (p_pos - neigh_pos) * d_const_surface_tension_gamma * initialMass * initialMass * splineCoefficient * (1.0f/dist);

	return force;
}

__device__
	inline float4 CalculateSurfaceCohesionForcePCISPHDevice(const uint dist_lut, const float dist, const float4 p_pos, const float4 neigh_pos, float neigh_weighted_vol)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	const float splineCoefficient = lutSplineSurfaceAdhesionTable[dist_lut];
	
	force = (p_pos - neigh_pos) * d_const_surface_adhesion_beta * initialMass * fluidRestDensity * neigh_weighted_vol * splineCoefficient * (1.0f/dist); 

	return force;	
}

__device__
	inline float4 CalculatePressureForce(uint dist_lut, float4 p_pos, float4 neigh_pos, float p_pressure, float p_inv_density, float neigh_pressure, float neigh_inv_density)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// sum up pressure force according to Monaghan
	float4 kernel_gradient = ( p_pos - neigh_pos ) * lutKernelPressureGradientTable[dist_lut];
	float press_grad = p_pressure * p_inv_density * p_inv_density + neigh_pressure * neigh_inv_density * neigh_inv_density;
	force = kernel_gradient * press_grad * initialMass2 * (-1.0f);			// Note: don't forget the force is negative

	return force;
}

__device__
	inline float4 CalculatePressureForcePCISPH(uint dist_lut, float4 p_pos, float4 neigh_pos, float p_pressure, float neigh_pressure)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// sum up pressure force according to Monaghan
	float4 kernel_gradient = ( p_pos - neigh_pos ) * lutKernelPressureGradientTable[dist_lut];
	float press_grad = (p_pressure + neigh_pressure) * invfluidRestDensity * invfluidRestDensity;
	force = kernel_gradient * press_grad * initialMass2 * (-1.0f);			// Note: don't forget the force is negative

	return force;
}

__device__
	inline float4 CalculateBoundaryFluidPressureForceDevice(uint dist_lut, float4 p_pos, float4 neigh_pos, float p_density, float p_corr_pressure, float weighted_vol)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float  p_inv_den  = 1.0f / p_density;  

	float4 kernel_gradient = ( p_pos - neigh_pos ) * lutKernelPressureGradientTable[dist_lut];
	float press_grad = p_corr_pressure * p_inv_den * p_inv_den;			// TODO: test if we could use p_inv_den * p_inv_den instead of invfluidRestDensity * invfluidRestDensity in here
	force = kernel_gradient * press_grad * initialMass *fluidRestDensity * weighted_vol * (-1.0f);		// Note: don't forget the force is negative

	return force;
}

// Rigid body dynamics

__device__ 
	inline float4 CalculateSpringForce_d(const float& dist, const float& overlap, const float4& r_ij)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	force.x = -1.0f * d_const_spring_coefficient * overlap * r_ij.x / dist;
	force.y = -1.0f * d_const_spring_coefficient * overlap * r_ij.y / dist;
	force.z = -1.0f * d_const_spring_coefficient * overlap * r_ij.z / dist;
	force.w = 0.0f;

	return force;
}

__device__ 
	inline float4 CalculateDampingForce_d(const float4& v_ij)
{
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	force.x = d_const_damping_coefficient * v_ij.x;
	force.y = d_const_damping_coefficient * v_ij.y;
	force.z = d_const_damping_coefficient * v_ij.z;
	force.w = 0.0f;

	return force;
}

__device__
	inline void PerformLinearStep_d(const int rigid_body_index, const float inv_mass, const float4& linear_momentum, const float& time_step, float4& pos, float4& vel)
{
	// v = p/m & p += vt
	vel = linear_momentum * inv_mass;
	pos += vel * time_step;
}

__device__
	inline void PerformAngularStep_d(const int rigid_body_index, const Matrix3x3_d inv_inertia_tensor_world, 
	const float4& angular_momentum, const float& time_step, float4& angular_vel, float4& quaternion)
{
	//update angular velocity : w = I^-1 * L
	angular_vel = MatrixVectorMul(inv_inertia_tensor_world, angular_momentum);

	float angular_speed = getLength(angular_vel);
	if (angular_speed > 0.0)
	{
		// rotation axis a = w/|w| 
		// Equation 7 in "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
		float4 rotationAxis = angular_vel * (1.0f/angular_speed);

		// theta = |w * dt|
		float rotationAngle = angular_speed*time_step;

		// dq = (cos(theta/2), a * sin(theta/2))
		float ds = cos(rotationAngle/2.0f);
		float dvx = rotationAxis.x*sin(rotationAngle/2.0f);
		float dvy = rotationAxis.y*sin(rotationAngle/2.0f);
		float dvz = rotationAxis.z*sin(rotationAngle/2.0f);

		float s = quaternion.x;
		float vx = quaternion.y;
		float vy = quaternion.z;
		float vz = quaternion.w;

		// q(t+dt) = dq x q(t)
		// Equation 8 in "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
		quaternion.x = s*ds - vx*dvx - vy*dvy - vz*dvz;
		quaternion.y = ds*vx + s*dvx + dvy*vz - dvz*vy;
		quaternion.z = ds*vy + s*dvy + dvz*vx - dvx*vz;
		quaternion.w = ds*vz + s*dvz + dvx*vy - dvy*vx;
	}
}

__device__
	inline void NormalizeQuaternion_d(float4& quaternion)
{
	float mag2 = quaternion.x * quaternion.x +
				 quaternion.y * quaternion.y +
				 quaternion.z * quaternion.z +
				 quaternion.w * quaternion.w;

	if (mag2!=0.0f && (fabs(mag2 - 1.0f) > 0.00001f)) {
		float mag = sqrtf(mag2);
		quaternion.x /= mag;
		quaternion.y /= mag;
		quaternion.z /= mag;
		quaternion.w /= mag;
	}
}

__device__
	inline void UpdateRotationMatrix_d(float4& quaternion, Matrix3x3_d& rotation_matrix)
{
	NormalizeQuaternion_d(quaternion);

	// then calculate rotation matrix from quaternion q = [s, v_x, v_y, v_z] 
	/************************************************************************/
	/*  1-2*v_y*v_y-2*v_z*v_z  2*v_x*v_y-2*s*v_z       2*v_x*v_z+2*s*v_y    */
	/*  2*v_x*v_y+2*s*v_z      1-2*v_x*v_x-2*v_z*v_z   2*v_y*v_z-2*s*v_x    */
	/*  2*v_x*v_z-2*s*v_y      2*v_y*v_z+2*s*v_x       1-2*v_x*v_x-2*v_y*v_y*/
	/************************************************************************/
	float w = quaternion.x;
	float x = quaternion.y;
	float y = quaternion.z;
	float z = quaternion.w;

	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float xy = x * y;
	float xz = x * z;
	float yz = y * z;
	float wx = w * x;
	float wy = w * y;
	float wz = w * z; 

	rotation_matrix.m_row[0].x = 1.0f-2.0f*(yy+zz);
	rotation_matrix.m_row[0].y = 2.0f*(xy-wz);
	rotation_matrix.m_row[0].z = 2.0f*(xz+wy);
	rotation_matrix.m_row[0].w = 0.0f;

	rotation_matrix.m_row[1].x = 2.0f*(xy+wz);
	rotation_matrix.m_row[1].y = 1.0f-2.0f*(xx+zz);
	rotation_matrix.m_row[1].z = 2.0f*(yz-wx);
	rotation_matrix.m_row[1].w = 0.0f;

	rotation_matrix.m_row[2].x = 2.0f*(xz-wy);
	rotation_matrix.m_row[2].y = 2.0f*(yz+wx);
	rotation_matrix.m_row[2].z = 1.0f-2.0f*(xx+yy);
	rotation_matrix.m_row[2].w = 0.0f;
}

__device__
	inline void ApplyRotationToParticle_d(const Matrix3x3_d& rotation_matrix, 
										  const float4& original_relative_pos, 
										  const float4& rb_pos, 
										  const float4& rb_vel, 
										  const float4& angular_vel, 
										  float4& p_pos, 
										  float4& p_vel)
{
	/*
	We apply the rotation by multiplying the initial relative position vector by the rotation matrix, 
	yielding a new relative position vector, and then add the rigid body position vector to this.
	This is slightly different from Equation 18 in "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3,
	which uses quaternions instead of matrix to update relative position.
	*/

	const float4& new_relative_pos = MatrixVectorMul(rotation_matrix, original_relative_pos);
	p_pos = rb_pos + new_relative_pos;
	p_vel = rb_vel + getCrossProduct(angular_vel, new_relative_pos);

}

__device__
	inline void UpdateInverseInertiaTensor_d(const int rigid_body_index, 
											 Matrix3x3_d rotation_matrix, 
											 Matrix3x3_d init_inv_inertia_tensor_local, 
											 Matrix3x3_d& inv_inertia_tensor_world
											 )
{

	// I^-1 = R * I_local^-1 * R^T
	inv_inertia_tensor_world = MatrixMul( MatrixMul(rotation_matrix, init_inv_inertia_tensor_local), getTranspose(rotation_matrix) );

}

__device__
	inline void CalculateForceTorque_d(const int rigid_body_index, 
								       const int num_particles, 
									   const float4* particle_force_array, 
									   const float4& rigid_body_pos, 
									   const int* rigid_particle_indices, 
									   float4& rb_force, 
									   float4& rb_torque)
{
	// calculate force and torque for each rigid body by accumulating child particle's contribution

	// iterate over all child particles, and accumulate its contribution
	for (int i = 0; i < num_particles; ++i)
	{
		const int p_index = rigid_particle_indices[i];
		const float4& p_force = particle_force_array[p_index];
		rb_force += p_force;
		float4 relative_pos = tex1Dfetch(texture_pos_zindex, p_index) - rigid_body_pos;	 // p_j - center_of_mass
		rb_torque += getCrossProduct(relative_pos, p_force);
	}
}

__device__
	inline void CalculateForceTorqueTwoWayCoupling_d(const int rigid_body_index, 
													 const int num_particles,
													 const float4& rigid_body_pos, 
													 const int* rigid_particle_indices, 
													 float4& rb_force, float4& rb_torque
													 )
{
	// calculate force and torque for each rigid body by accumulating child particle's contribution

	// iterate over all child particles, and accumulate its contribution
	for (int i = 0; i < num_particles; ++i)
	{
		const int p_index = rigid_particle_indices[i];
		const float4& p_force = tex1Dfetch(texture_static_force, p_index);
		rb_force += p_force;
		float4 relative_pos = tex1Dfetch(texture_pos_zindex, p_index) - rigid_body_pos;	 // p_j - center_of_mass
		rb_torque += getCrossProduct(relative_pos, p_force);
	}
}

__device__
	inline void Clamp_d(const float& terminal_value, float4& vec)
{
	if (vec.x > 0.0f)
		vec.x = fminf(vec.x, terminal_value);
	else
		vec.x = fmaxf(vec.x, -terminal_value);

	if (vec.y > 0.0f)
		vec.y = fminf(vec.y, terminal_value);
	else
		vec.y = fmaxf(vec.y, -terminal_value);

	if (vec.z > 0.0f)
		vec.z = fminf(vec.z, terminal_value);
	else
		vec.z = fmaxf(vec.z, -terminal_value);

	if (vec.w > 0.0f)
		vec.w = fminf(vec.w, terminal_value);
	else
		vec.w = fmaxf(vec.w, -terminal_value);
}

__device__
	inline void UpdateMomenta_d(const int rigid_body_index, 
							    const float time_step, 
								const float4& rb_force, 
								const float4& rb_torque, 
								const float& rb_mass, 
								float4& rb_linear_momentum, 
								float4& rb_angular_momentum
								)
{
 rb_linear_momentum += rb_force * time_step;
 
 rb_angular_momentum += rb_torque * time_step;

 /*
  We check whether the absolute value of the linear momentum in each direction is greater than a maximum value
  calculated from a user defined terminal momentum. We clamp the momentum between the range
  [−maximum momentum, maximum momentum]. We do this to prevent the rigid bodies from accelerating to unrealistically great speeds when
  they fall for a long time. from paper "Real-Time Rigid Body Interactions" P38
 */
 // clamp operation
 const float terminal_momentum = d_const_terminal_speed * rb_mass;
 Clamp_d(terminal_momentum, rb_linear_momentum);

}

#endif // GPU_UNIFIED_PARTICLES_KERNEL_H_