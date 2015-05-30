#ifndef GPU_UNIFIED_PARTICLES_H_
#define GPU_UNIFIED_PARTICLES_H_

#include "host_defines.h"
#include "cutil.h"
#include "vector_types.h"
#include <limits>
#include <stdint.h>
#include "UnifiedParticle.h"
#include "UnifiedConstants.h"
#include "UnifiedMathCUDA.cuh"
#include "System/Profiling.h"

#define SYNCIT __syncthreads()

typedef unsigned int uint;
typedef unsigned short ushort;
/*
	Trick!!! 
	NOTE: Here we use this to avoid deep copy when using int* rigid_particle_indices;	in struct RigidBody_GPU
	deep copy is a little complex in current cuda mode. Well, Unified Memory in CUDA 6/7 will make it much easier
	see http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/ 
*/
#define MAX_NUM_RIGID_PARTICLES 15000	    

// For rigid body, we use AOS
typedef struct __align__(16)
{
	Matrix3x3_d  rotation_matrix;			// TODO: we would like to use quaternion to repalce rotation matrix    Note: the num of elements in rotation matrix is constant = 9
	Matrix3x3_d  inv_inertia_local;			// TODO: this could be precomputed & stored in texture mem, bind once and unbind before existing program
	Matrix3x3_d  inv_inertia_world;	
	float4		 pos;						// rigid body's center of mass
	float4		 vel;
	float4		 angular_velocity;
	float4       force;
	float4		 torque;
	float4	     linear_momentum;
	float4		 angular_momentum;
	float4	     quaternion;	
	float4		 replacement;
	int			 rigid_particle_indices[MAX_NUM_RIGID_PARTICLES];		  // Note: We need a DEEP COPY in here if we use int* rigid_particle_indices;
	float		 mass;
	int			 num_particles;				// record the number of child particles
}RigidBody_GPU;

// Particle splitted into 3 different structures
// CUDA texture has to be one of the basic data types
// float4 implicity __aligned(16)__
typedef struct
{
    uint		 demoScene;
    
    // "finalParticleCount" introduced to keep track
    // of new particles added every frame in new
    // simulation
	
	uint		 particleCount;
	uint		 particleCountRounded;
	uint		 finalParticleCount;
	uint		 finalParticleCountRounded;
	uint		 filledBlocks;
	
	uint		 dimX;
	uint		 dimY;
	uint		 dimZ;
	uint		 maxLength;
	
    uint		 filteredParticleCount;
	
	float		 globalSupportRadius;
	float		 distToCenterMassCutoff;
	float		 gradientVelCutoff;
	float		 particleRadius;
	float3		 scales;
	uint		 grid_resolution;
	uint		 block_size;
	uint		 currIndex;			// Index containing recent most particles, 0 or 1
	uint		 currType;			// Index containing recent most particles, 0 or 1 for type information
	uint		 currActiveType;	// Index containing recent most particles, 0 or 1 for type information 

	float	     rb_spring;
	float		 rb_spring_boundary;
	float		 rb_damping;
	float		 rb_terminalSpeed;

	float	     surface_tension_gamma;
	float		 surface_adhesion_beta;
	
	uint		 lutSize;
	float		 kernelSelf;
	float		 initialMass;
	float		 fluidRestDensity;
	float        gamma;
	float		 fluidGasConstant;
	float        fluidGasConstantWCSPH;
	float		 fluidViscosityConstant;
	float		 fluidViscosityConstant_tube;
	float		 gravityConstant;
	bool		 addBoundaryForce;
	float3		 minCollisionBox;
	float3		 maxCollisionBox;
	float3		 minBoundingBox;
	float3		 maxBoundingBox;
	float3		 minContainerBox;
	float3		 maxContainerBox;
	float3		 zindexStartingVec;
	float		 boxLength;
	float		 boxHeigth;
	float		 boxWidth;
	float		 wallX;
	float3		 pipePoint1;
	float3		 pipePoint2;
	float		 pipeRadius;
	float		 forceDistance;
	float		 maxBoundaryForce;
	float		 deltaT;
	float        deltaTWCSPH;
	float		 spacing;
	float		 particleRenderingSize;
	unsigned int num_rigid_bodies;
	
	// For particles, we use SOA
	// Particle Attributes
	float4*		 d_pos_zindex[2];
	float4*		 d_rigid_particle_relative_pos[2];			// each rigid particle has one relative_pos, currently we store the last element as 1 if it's a rigid particle, otherwise set it as 0 TODO: to be optimized
	float4*		 d_vel[2];
	float4*		 d_force;									// we need this in rigid body force calculation
	float4*		 d_filtered_pos_zindex;						// What does this mean?
	float4*		 d_smoothcolor_normal;
	float*		 d_pressure;
	float*		 d_density;
	float*		 d_weighted_volume;
	int*		 d_type[2];									// 0: RIGID_PARTICLE 1: SOFT_PARTICLE 2: LIQUID_PARTICLE 3: GRANULAR_PARTICLE 4: CLOTH_PARTICLE 5: SMOKE_PARTICLE 6: FROZEN_PARTICLE
	int*		 d_active_type[2];							// 0: active particle	1: semi-active particle	  2: passive particle
	int*		 d_parent_rb[2];							// parent rigid body index, each particle stores one permanent parent rigid body index information
	int*		 d_order_in_child_particles_array[2];

	// PCISPH related Attributes
	float4*       d_predictedPos;
	float4*       d_predictedVel;							// TODO: delete
	float4*       d_correctionPressureForce;				// total force = d_static_force + d_dynamic_boundary_force + d_correctionPressureForce
	float4*		  d_static_force;							// static boundary force + viscosity force + gravitational force + user-defined forces
	float4*		  d_dynamic_boundary_force;					// dynamic boundary force (boundary-fluid pressure & friction force)
	float*        d_correctionPressure;						
	float*		  d_previous_correctionPressure;			// only used in CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH
	float*        d_densityError;
	float*        d_predictedDensity;
	float*		  d_previous_predicted_density;				// only used in CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH
	float*		  d_max_predicted_density_array; 
	float*		  d_max_predicted_density_value;
	float		  param_density_error_factor;
	float		  param_max_density_error_allowed;
	uint          param_num_threads_custom_reduction;
	uint		  param_num_blocks_custom_reduction;
	int			  param_min_loops;
	int			  param_max_loops;

	// Particle based rigid body related Attributes
	RigidBody_GPU* d_rigid_bodies;

	/*
		We check whether the absolute value of the linear momentum in each direction is greater than a maximum value
		calculated from a user defined terminal momentum. We clamp the momentum between the range
		[−maximum momentum, maximum momentum]. We do this to prevent the rigid bodies from accelerating to unrealistically great speeds when
		they fall for a long time. from paper "Real-Time Rigid Body Interactions" P38
	*/
	float*		  d_rb_terminal_momentum;


	// particlesKVPair_d[0] is both the input and output array - data will be sorted
	// particlesKVPair_d[1] is just an additional array to allow ping pong computation
	uint*		 particlesKVPair_d[2];			// (key,value) = (zindex, i) i -> particle position Also for the "blocks maintenance"
	
	uint*		 tmpArray;

	uint		 m_posVbo[2];		// ping-pong vertex buffer object for particle positions
	
#ifdef DUMP_PARTICLES_TO_FILE
	uint32_t*	 cpuMem1;
	float*		 cpuMem2;
#endif

} dataPointers;

static const int MAX_INDICES_PER_CUBE = 40;
static const int MAX_THREADS_PER_BLOCK_SPH = 64;
static const int LUTSIZE = 300;

__global__ 
	void CopySortedParticleValuesKernel( unsigned int particleCount, 
										 float4* array_pos_, 										
										 float4* array_vel_, 										
										 int* array_type_, 
										 unsigned int* particlesKVPair_d );

__global__ 
	void CopySortedParticleValuesKernelApproximate( unsigned int particleCount, 
	float4* array_pos_, 										
	float4* array_vel_, 										
	int* array_active_type_, 
	unsigned int* particlesKVPair_d );

__global__ 
	void CopySortedParticleValuesKernelRigidBodySPH(unsigned int particleCount, 
													float4* array_pos_, 										
													float4* array_vel_, 
													float4* array_relative_pos_,
													int* array_type_, 
													int* array_parent_rb,
													int* array_order_in_child_particles_array,
													unsigned int* particlesKVPair_d );

__global__ 
	void CopySortedParticleValuesKernelStaticBoundariesPCISPH(unsigned int particleCount, 
	float4* array_pos_, 										
	float4* array_vel_, 
	int* array_type_, 
	float* array_previous_corr_pressure,
	float* array_predicted_density,
	unsigned int* particlesKVPair_d );

__global__ 
	void CopySortedParticleValuesKernelRigidBodyPCISPH(unsigned int particleCount, 
													float4* array_pos_, 										
													float4* array_vel_, 
													float4* array_relative_pos_,
													int* array_type_, 
													int* array_parent_rb,
													int* array_order_in_child_particles_array,
													float* array_previous_corr_pressure,
													float* array_predicted_density,
													unsigned int* particlesKVPair_d );
	
__global__
void ClearIndexArrays(uint* indices);
            
__global__
void CalculateBlocksKernel(uint* indices1, uint* indices2, uint* tmpArray);

__global__
void RedistributeBlocksKernel(uint* indices1, uint* indices2);
					
__global__
void CalculateDensitiesInBlocksKernelSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
									uint* indices1, uint* indices2 );

__global__
	void CalculateDensitiesInBlocksKernelSPHApproximate(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
	void MarkMoreActiveParticlesStateInBlocksKernelSPH(int* array_particle_state_, uint* indices1, uint* indices2 );

__global__
	void MarkAllRemainingParticlesStateInBlocksKernelSPH(int* array_particle_state_, uint* indices1, uint* indices2 );

__global__
	void FindActiveParticlesInBlocksKernelSPH(int* array_particle_state_, uint* indices1, uint* indices2 );

__global__
	void CalculateDensitiesInBlocksKernelWCSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
	void CalculateDensitiesInBlocksKernelWCSPHApproximate(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
void CalculateForcesInBlocksKernelSPH( float4* array_pos_, float4* array_vel_,
 							float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

__global__
	void CalculateForcesInBlocksKernelSPHApproximate( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

__global__
	void CalculateForcesInBlocksKernelWCSPH( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

__global__
	void CalculateForcesInBlocksKernelWCSPHApproximate( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

__global__
void ExtractSurfaceParticlesKernel( float4 *array_filtered_pos_zindex, uint* indices1, uint* indices2 );

__global__
void ExtractSurfaceParticlesKernel2( float4 *array_filtered_pos_zindex, uint* indices1, uint* indices2 );

__global__
void FillZindicesKernel(const float4* array_pos_, uint* indices);

// PCISPH methods
__global__
	void CalculateExternalForcesInBlocksKernelPCISPH( float4* static_external_force_, float4* corr_pressure_force_pressure_, float* correction_pressure_, uint* indices1, uint* indices2 );

__global__
	void PredictPositionAndVelocityBlocksKernelPCISPH(float4* predicted_pos_);

__global__
	void ComputePredictedDensityAndPressureBlocksKernelPCISPH(float* density_error_, float* predicted_density_, float* corr_pressure_, uint* indices1, uint* indices2 );

__global__
	void ComputePredictedDensityAndPressureBlocksKernelPCISPHWallWeight(float* density_error_, float* predicted_density_, float* corr_pressure_, uint* indices1, uint* indices2 );

__global__
	void ComputeCorrectivePressureForceBlocksKernelPCISPH(float4* corr_pressure_force_, uint* indices1, uint* indices2 );

__global__
	void ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPH(float4* corr_pressure_force_, uint* indices1, uint* indices2 );

__global__
	void ComputeCorrectivePressureForceBlocksKernelPCISPHTwoWayCoupling(float4* corr_pressure_force_, uint* indices1, uint* indices2 );

__global__
	void ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPHTwoWayCoupling(float4* corr_pressure_force_, uint* indices1, uint* indices2 );

__global__
	void TimeIntegrationBlocksKernelPCISPHPureFluid( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 );

__global__
	void TimeIntegrationBlocksKernelStaticBoundariesPCISPH( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 );

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep1( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 );

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep2( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 );

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep3( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 );

// Wall Weight Method
__global__
	void CalculateCorrectedDensityPressureInBlocksKernelSPHWallWeight(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
	void CalculateCorrectedDensityPressureInBlocksKernelWCSPHWallWeight(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

// Versatile coupling method
__global__
	void CalculateCorrectedDensitiesInBlocksKernelSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
	void CalculateCorrectedDensitiesInBlocksKernelWCSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 );

__global__
	void CalculateWeightedVolumeInBlocksKernel(float* array_weighted_volume_, uint* indices1, uint* indices2 );

__global__
	void CalculateForcesPerParticleInBlocksKernelSPHVersatileCoupling( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

__global__
	void CalculateForcesPerParticleInBlocksKernelWCSPHVersatileCoupling( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 );

// Particle based rigid body dynamics
__global__
	void CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH( float4* static_force_, 
																		uint* indices1, 
																		uint* indices2 );

__global__
	void CalculateExternalForcesStaticBoundariesInBlocksKernelPCISPH( float4* static_force_, uint* indices1, uint* indices2 );

__global__
	void CalculateExternalForcesWithoutBoundaryForceRigidFluidCouplingInBlocksKernelPCISPH( float4* static_force_, 
	uint* indices1, 
	uint* indices2 );

__global__
	void UpdateRigidParticleIndicesKernel(RigidBody_GPU* rigid_bodies_gpu);

__global__
	void CalculateForcesBetweenRigidFluidParticlesInBlocksKernelSPH( float4* array_force, 
																/*float4* array_smoothcolor_normal,*/ 
																uint* indices1, 
																uint* indices2 );

__global__
	void CalculateForcesBetweenRigidParticlesInBlocksKernelWCSPH(float4* array_force,
																 /*float4* array_smoothcolor_normal,*/
																 uint* indices1, 
																 uint* indices2 
																 );

__global__ 
	void RigidBodyIntegrationKernel(const int			currIndex,
									const float4* 		particle_force,
									RigidBody_GPU*		rigid_bodies_gpu								
									);

__global__ 
	void RigidBodyIntegrationTwoWayCouplingKernel(const int			currIndex,
												  RigidBody_GPU*		rigid_bodies_gpu								
												  );

__global__
	void SynRigidParticlesKernel(const int currIndex,
								 RigidBody_GPU*		rigid_bodies_gpu,
								 float4*			p_pos_array,
								 float4*			p_vel_array);

__global__
	void CalculateCorrectedForcePerParticleKernel(float4* corrected_force_array);

__global__
	void UpdateLiquidParticleKernelSPH(const int		currIndex, 
									const float4*	p_force_array,
									float4*			p_pos_array,
									float4*			p_vel_array
									);

__global__
	void UpdateLiquidParticleKernelPCISPH(const int		currIndex, 
									float4*			p_pos_array,
									float4*			p_vel_array
									);

__global__
	void ComputeCorrectiveBoundaryFluidForceBlocksKernelPCISPH(float4* dynamic_boundary_force_, uint* indices1, uint* indices2 );

/*======================================================================================================*/

extern "C" {
	//void ExtractSurfaceParticles(dataPointers& dptr, int blockCount);
	void MallocDeviceArraysParticles(dataPointers& dptr);
	void MallocDeviceArraysRigidBody(dataPointers& dptr, std::vector<RigidBody*>& rigidbodies_h);
	void SetDeviceConstants(dataPointers& dptr);
	void FreeDeviceArrays(dataPointers& dptr);
	void CopyIndicesDeviceToHost(dataPointers& dptr,uint*& indices);
	void CopyZIndicesKVPairHostToDevice(dataPointers& dptr,std::vector<UnifiedParticle>& particles_h);
	void CopyParticleDataHostToDevice(dataPointers& dptr,std::vector<UnifiedParticle>& particles_h);
	void CopyRigidBodyDataHostToDevice(const std::vector<RigidBody*>& rigidbodies_h, dataPointers& dptr);
	void CopyParticlesDeviceToHost(dataPointers& dptr,std::vector<UnifiedParticle>& particles_h, float*& p);

#ifdef OUTPUT_GPU_RIGID_BODY_INFO

	void UpdateRigidBodyPovInfo(dataPointers& dptr, vector<RigidBodyOfflineRenderingInfo>& rigid_body_info_);

#endif
	
	//void CopyAllParticleLayersHostToDevice( dataPointers& dptr, float* pos_zindex_Host, float* vel_pressure_Host, unsigned int* zindex_Host, uint count );
	//void CopyParticleLayerHostToDevice(dataPointers& dptr, uint count);
	//void CopyParticleLayerHostToDevice2(dataPointers& dptr, uint count, float* pos_zindex_Host, float* vel_pressure_Host, unsigned int* zindex_Host);
	
	void CreateZIndexTexture( unsigned int* A );
	void CreateLutKernelM4Texture( float* A, const int lutSize );
	void CreateLutKernelPressureGradientTexture( float* A, const int lutSize );
	void CreateLutKernelViscosityLapTexture( float* A, const int lutSize );
	void CreateLutSplineSurfaceTensionTexture( float* A, const int lutSize );
	void CreateLutSplineSurfaceAdhesionTexture( float* A, const int lutSize );

	
	// Setting global variables
	void SetParticleCount( const unsigned int& count );
	void SetRigidBodyCount(const unsigned int& num_rigids);
	void SetGlobalSupportRadius( const float& rad );
	void SetDistToCenterMassCutoffApproximate( const float& cutoff);
	void SetVelCutoffApproximate( const float& cutoff);
	void SetParticleRadius(const float& radius);
	void SetTerminalSpeed(const float& terminal);
	void SetRigidBodyCoefficients(const float& spring, const float& spring_boundary, const float& damp);
	void SetSurfaceTensionAdhesionCoefficients(const float& surface_tension_gamma, const float& surface_adhesion_beta);
	void SetGridResolution( const unsigned int& grid_resolution );
	void SetBlockSize( const unsigned int& block_Size );
	void SetLutSize( const unsigned int& lut_size );
	void SetKernelSelf( const float& kernelSelf );
	void SetInitialMass( const float& initialMass );
	void SetFluidRestDensity( const float& fluidRestDensity );
	void SetFluidRestVolume( const float& initialMass, const float& fluidRestDensity);
	void SetZindexStartingVec( const float3& zindexStartingVec_);
	void SetGamma(const float& gamme);
	void SetFluidGasConstant( const float& fluidGasConstant, const float& fluidGasConstantWCSPH );
	void SetDensityErrorFactorPCISPH( const float& denErrorFactor_);
	void SetBoundaryConstants( const float& forceDistance, const float& maxBoundaryForce, const float3& minB, const float3& maxB, bool &add_boundary_force_);
	void SetBoundingBox( const float3& minB, const float3& maxB );
	void SetRealContainerValues(const float& length, const float& height, const float& width, const float3& minB, const float3& maxB);
	void SetDeltaT( const float& deltaT, const float& deltaTWCSPH );
	void SetScales( const float3& scale );
	void SetGravityConstant( const float& gravityConstant );
	void SetInitialMassIntoGravityRatio( const float& initialMass, const float& gravityConstant );
	void SetFluidViscosityConstant( const float& fluidViscosityConstant );
	void SetFluidViscosityConstantTube(float& fluidViscosityConstant_tube);
	void SetSpacing( const float& spacing );
	void SetParticleRenderingSize(const float& pradius);
	void SetOtherConstants();
	void CreateLutKernelM4( float* A, const int& lutSize );
	void CreateLutKernelPressureGradient( float* A, const int& lutSize );
	void CreateLutKernelViscosityLap( float* A, const int& lutSize );
	void CreateLutSplineSurfaceTension( float* A, const int& lutSize );
	void CreateLutSplineSurfaceAdhesion( float* A, const int& lutSize );

	void SetPipePoints(const float3& p1, const float3& p2);
	void SetPipeRadius(float& pipeRadius);
	void SetWallConstant(const float& wallX_);
	void SetGridDimensions(uint& dimX_, uint& dimY_, uint& dimZ_);
	void SetMaximumArrayLength(uint& arrL_);
	
	uint CopyParticlesCUDA( float* pos_zindex, float* vel, float* corr_pressure, float* predicted_density, int* particleType, int* activeType, dataPointers& dptr );

	// CUDA/Graphics Interoperability
	void RegisterGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void UnregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void* MapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void UnmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

	// SPH Methods
	void ZindexSorting(int currIndex, int currType, dataPointers& dptr);
	void ZindexSortingApproximate(int currIndex, int currActiveType, dataPointers& dptr);
	void BlockGeneration(dataPointers& dptr, int& x, int& y);
	void DensityPressureComputation(int currIndex, dataPointers& dptr, int& y);
	void DensityPressureComputationApproximate(int currIndex, dataPointers& dptr, int& y);
	void FindActiveParticles(int currActiveType, dataPointers& dptr, int& y);
	void MarkMoreActiveParticlesState(int currIndex, dataPointers& dptr, int& y);
	void MarkAllRemainingParticlesState(int currActiveType, dataPointers& dptr, int& y);
	void ForceComputation(int currIndex, dataPointers& dptr, int& y);
	void ForceComputationApproximate(int currIndex, dataPointers& dptr, int& y);
	void ExtractSurface(dataPointers& dptr, int& x);
	void CalculateZindex(int currIndex, dataPointers& dptr);
	void UnbindTexturesSPH();
	void UnbindTexturesSPHApproximate();
	void UnbindTexturesPCISPH();
	void UnbindTextureRigidFluidSPH();
	void UnbindTextureRigidFluidPCISPH();
	void PingPongScheme(dataPointers& dptr);
	void PingPongSchemeApproximate(dataPointers& dptr);
	void PingPongSchemeRigidBody(dataPointers& dptr);
	void ParticlePhysicsOnDeviceSPH(dataPointers& dptr);
	void ParticlePhysicsOnDeviceSPHApproximate(dataPointers& dptr);

	// PCISPH Methods
	void ParticlePhysicsOnDevicePCISPH(dataPointers& dptr);
	void CalculateExternalForcesPcisphCUDA(int currIndex, dataPointers& dptr, int& y);
	void PredictionCorrectionStepPcisphCUDA(dataPointers& dptr, int& y); 
	void PredictPositionAndVelocityPcisphCUDA(dataPointers& dptr, int& y);
	void ComputePredictedDensityAndPressurePcisphCUDA(dataPointers& dptr, int& y);
	void ComputeCorrectivePressureForcePcisphCUDA(dataPointers& dptr, int& y);
	void TimeIntegrationPcisphCUDAPureFluid(int currIndex, dataPointers& dptr, int& y);
	void GetMaxPredictedDensityCUDA(dataPointers& dptr, float& max_predicted_density);

	// boundary handling method from Ihmsen's paper "Boundary handling and adaptive time-stepping for PCISPH"
	void ParticlePhysicsOnDevicePCISPHIhmsen2010(dataPointers& dptr);
	void TimeIntegrationPcisphCUDAIhmsen2010Method(int currIndex, dataPointers& dptr, int& y); 

	// static complex boundary handling method
	void CalculateExternalForcesStaticBoundariesPcisphCUDA(int currIndex, dataPointers& dptr, int& y);
	void TimeIntegrationStaticBoundariesPcisphCUDA(int currIndex, dataPointers& dptr, int& y);
	void ZindexSortingStaticBoundariesPCISPH(int currIndex, int currType, dataPointers& dptr);
	void ParticlePhysicsOnDeviceFluidStaticBoundariesPCISPH(dataPointers& dptr);

	// Wall Weight Method
	void CorrectedDensityPressureComputationWallWeight(int currIndex, dataPointers& dptr, int& y);

	// versatile coupling method
	void ForceComputationVersatileCoupling(int currIndex, dataPointers& dptr, int& y);
	void CalculateWeightedVolume(int currIndex, dataPointers& dptr, int& y);
	void CorrectedDensityPressureComputation(int currIndex, dataPointers& dptr, int& y);
	void PredictionCorrectionStepPcisphCUDAVersatileCoupling(dataPointers& dptr, int& y);
	void ComputeCorrectivePressureBoundaryForcePcisphCUDA(dataPointers& dptr, int& y);

	// Rigid Body Dynamics
	void ZindexSortingRigidBodySPH(int currIndex, dataPointers& dptr);
	void ZindexSortingRigidBodyPCISPH(int currIndex, dataPointers& dptr);
	void ParticlePhysicsOnDeviceFluidRigidCouplingSPH(dataPointers& dptr);
	void UpdateRigidParticleIndicesArray(dataPointers& dptr);
	void RigidBodyIntegration(int currIndex, dataPointers& dptr);
	void SynRigidParticlesDevice(int currIndex, dataPointers& dptr);
	void CalculateCorrectedForcePerParticle(dataPointers& dptr);
	void UpdateLiquidParticlePosVelSphGPU(int currIndex, dataPointers& dptr);
	void CalculateZindexRigidBody(int currIndex, dataPointers& dptr);
	void ForceComputationRigidFluidParticles(int currIndex, dataPointers& dptr, int& y);

	// Rigid fluid coupling method for PCISPH
	void ParticlePhysicsOnDeviceFluidRigidCouplingPCISPH(dataPointers& dptr);
	void CalculateExternalForcesFluidRigidCouplingPcisphCUDA(int currIndex, dataPointers& dptr, int& y);
	void CalculateExternalForcesWithoutBoundaryForceFluidRigidCouplingPcisphCUDA(int currIndex, dataPointers& dptr, int& y);
	void UpdateLiquidParticlePosVelPCISPH(int currIndex, dataPointers& dptr);
	void RigidBodyIntegrationTwoWayCoupling(int currIndex, dataPointers& dptr);
	void ComputeCorrectiveBoundaryFluidForcePcisphCUDA(dataPointers& dptr, int& y);
}

template <class T> 
void GetReductionMaxArray(uint size, uint threads, uint blocks, T *d_idata, T *d_odata);

#endif	// GPU_UNIFIED_PARTICLES_H_
