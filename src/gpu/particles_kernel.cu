#include <iostream>
#include "particles_kernel.cuh"
#include "UnifiedMathCUDA.cuh"

//=====================================================================
//                      CUDA TEXTURES & ARRAYS
//=====================================================================

texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_pos_zindex;
texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_relative_pos;
texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_predicted_pos;
texture<float4,			cudaTextureType1D,	cudaReadModeElementType> texture_vel;
texture<float4,         cudaTextureType1D,  cudaReadModeElementType> texture_static_force;
texture<float4,         cudaTextureType1D,  cudaReadModeElementType> texture_dynamic_boundary_force;
texture<float4,         cudaTextureType1D,  cudaReadModeElementType> texture_corr_pressure_force;
texture<int,			cudaTextureType1D,	cudaReadModeElementType> texture_type;
texture<int,			cudaTextureType1D,	cudaReadModeElementType> texture_active_type;
texture<int,			cudaTextureType1D,	cudaReadModeElementType> texture_parent_rb;
texture<int,			cudaTextureType1D,	cudaReadModeElementType> texture_order_in_child_particles_array;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_density;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_pressure;
texture<float,			cudaTextureType1D,  cudaReadModeElementType> texture_weighted_volume;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_corr_pressure;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_previous_corr_pressure;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_density_error;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_predicted_density;
texture<float,			cudaTextureType1D,	cudaReadModeElementType> texture_previous_predicted_density;
texture<unsigned int,	cudaTextureType2D,	cudaReadModeElementType> texture_zindex_array;

cudaArray* array_zindex;

//=====================================================================
//                      GRID SIZES & C++ VARIABLES
//=====================================================================

int num_blocks;
int gridsize_indices;

#if defined(SPH_PROFILING) || defined(SPH_PROFILING_VERBOSE)
unsigned int frameCounter_                        = 0;
float        simulationTimeCounter_               = 0.0f;
float        surfaceExtractionTimeCounter_        = 0.0f;
float		 parallelSortingTimeCounter_          = 0.0f;
float		 blockGenerationTimeCounter_          = 0.0f;
float		 externalForceCalculationTimeCounter_ = 0.0f;
float		 pciLoopTimeCounter_                  = 0.0f;
float		 timeIntegrationTimeCounter_          = 0.0f;
float		 indexCalculationTimeCounter_         = 0.0f;
#endif

//=====================================================================
//                      Definitions of Device Variables
//=====================================================================

__device__ unsigned int filteredCount;
__device__ unsigned int blockCount;

//=====================================================================
//                      Definitions of Constant Variables
//=====================================================================

__constant__ static unsigned int xMASK;
__constant__ static unsigned int yMASK;
__constant__ static unsigned int zMASK;


__constant__ static unsigned int PCOUNT;			// number of particles in simulation
__constant__ static unsigned int PCOUNTROUNDED;		// number of particles in simulation rounded
__constant__ static unsigned int RIGIDBODYCOUNT;	// number of rigid bodies in simulation

__constant__ static float globalSupportRadius;
__constant__ static float invglobalSupportRadius;
__constant__ static float globalSupportRadius2;
__constant__ static float d_const_particle_radius;
__constant__ static float d_const_dist_to_center_mass_cutoff;
__constant__ static float d_const_vel_cutoff;
__constant__ static float d_const_terminal_speed;
__constant__ static float fluidRestDensity;
__constant__ static float invfluidRestDensity;
__constant__ static float fluidRestVolume;
__constant__ static float gamma;
__constant__ static float fluidGasConstant;
__constant__ static float fluidGasConstantWCSPH;
__constant__ static float fluidViscosityConstant;
__constant__ static float fluidViscosityConstant_tube;
__constant__ static float gravityConstant;
__constant__ static float3 scales;
__constant__ static float3 zindexStartingVec;
__constant__ static unsigned int gridResolution;
__constant__ static unsigned int block_size;
__constant__ static float invblock_size;
__constant__ static unsigned int block_size3;
__constant__ static unsigned int lutSize;
__constant__ static float invlutSize;
//__constant__ static float globalSupportRadiusByLutSize;
__constant__ static float kernelSelf;

// PCISPH
__constant__ static float densityErrorFactor;

__constant__ static float initialMass;
__constant__ static float initialMass2;
__constant__ static float invinitialMass;
__constant__ static float initialMassIntoGravityConstant;

__constant__ static int intMax;

__constant__ static bool addBoundaryForce;
// Collision Box
__constant__ static float3 minCollisionBox;
__constant__ static float3 maxCollisionBox;
// Virtual Z-index Bounding Box
__constant__ static float3 minBoundingBox;
__constant__ static float3 maxBoundingBox;
// Real Container Box
__constant__ static float3 d_const_min_container_box;
__constant__ static float3 d_const_max_container_box;

// Wall Weight Function Method
__constant__ static float d_const_box_length;
__constant__ static float d_const_box_height;
__constant__ static float d_const_box_width;

__constant__ static float wallX;

__constant__ unsigned int dimX;
__constant__ unsigned int dimY;
__constant__ unsigned int dimZ;
__constant__ unsigned int total_blocks;
__constant__ unsigned int maxArrayLength;

__constant__ static float forceDistance;
__constant__ static float invforceDistance;
__constant__ float deltaT;
__constant__ static float deltaTWCSPH;
__constant__ static float maxBoundaryForce;
__constant__ static float spacing;
__constant__ static float lutKernelM4Table[LUTSIZE];
__constant__ static float lutKernelPressureGradientTable[LUTSIZE];
__constant__ static float lutKernelViscosityLapTable[LUTSIZE];
__constant__ static float lutSplineSurfaceTensionTable[LUTSIZE];
__constant__ static float lutSplineSurfaceAdhesionTable[LUTSIZE];


__constant__ static float CENTER_OF_MASS_THRESHOLD;
__constant__ static unsigned int N_NEIGHBORS_THRESHOLD;

// rigid body
__constant__ static float d_const_spring_coefficient;
__constant__ static float d_const_spring_boundary_coefficient;
__constant__ static float d_const_damping_coefficient;

// surface tension & Adhesion
__constant__ static float d_const_surface_tension_gamma;
__constant__ static float d_const_surface_adhesion_beta;

// Pipe points for second demo set up
__constant__ static float3 pipePoint1;
__constant__ static float3 pipePoint2;
__constant__ static float pipeLength;
__constant__ static float invpipeLength;
__constant__ static float pipeLength2;
__constant__ static float pipeRadius;

__constant__ static float particleRenderingSize; 

// mod without divide, works on values from 0 upto 2m
#define WRAP(x,m) (((x)<m) ? (x) : (x-m))

//=======================================================================
// Functions to set global from host to device
//=======================================================================
extern "C"
{
	void SetParticleCount( const unsigned int& count_ )
	{
		unsigned int countRounded = count_;
		if( countRounded % MAX_THREADS_PER_BLOCK_SPH )
			countRounded += (MAX_THREADS_PER_BLOCK_SPH - countRounded % MAX_THREADS_PER_BLOCK_SPH );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( PCOUNT, &count_, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( PCOUNTROUNDED, &countRounded, sizeof(unsigned int) ) );

		num_blocks = ceil((float)countRounded/MAX_THREADS_PER_BLOCK_SPH);

		std::cout << "Setting particle count " << count_ << " Rounded " << countRounded << std::endl;
	}

	void SetRigidBodyCount(const unsigned int& num_rigids)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(RIGIDBODYCOUNT, &num_rigids, sizeof(unsigned int) ) );
		std::cout << "Setting rigid body count " << num_rigids << std::endl;
	}

	void SetGlobalSupportRadius( const float& rad_ )
	{
		float globalSupportRadius_n = rad_ ;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( globalSupportRadius, &globalSupportRadius_n, sizeof(float) ) );
		globalSupportRadius_n = rad_ * rad_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( globalSupportRadius2, &globalSupportRadius_n, sizeof(float) ) );

		float f = 1.0 / rad_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invglobalSupportRadius, &f, sizeof(float) ) );

		std::cout << "Setting global support radius " << rad_ << std::endl;
	}

	void SetDistToCenterMassCutoffApproximate(const float& cutoff)
	{
		float distToCenterMassCutoff = cutoff;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_dist_to_center_mass_cutoff, &distToCenterMassCutoff, sizeof(float) ) );
		std::cout << "Setting Gradient Color Cutoff " << cutoff << std::endl;
	}

	void SetVelCutoffApproximate( const float& cutoff)
	{
		float vel_cutoff = cutoff;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_vel_cutoff, &vel_cutoff, sizeof(float) ) );
		std::cout << "Setting Velocity Cut off " << cutoff << std::endl;
	}

	void SetParticleRadius(const float& radius)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_particle_radius, &radius, sizeof(float) ) );
		std::cout << "Setting particle radius " << radius << std::endl;
	}

	void SetTerminalSpeed(const float& terminal)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_terminal_speed, &terminal, sizeof(float) ) );
		std::cout << "Setting terminal momentum " << terminal << std::endl;
	}

	void SetRigidBodyCoefficients(const float& spring, const float& spring_boundary, const float& damping)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_spring_coefficient,			&spring,			sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_spring_boundary_coefficient, &spring_boundary,	sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_damping_coefficient,			&damping,			sizeof(float) ) );

		std::cout << "Setting rigid body spring coefficient: " <<  spring			<< std::endl;
		std::cout << "Setting rigid body spring coefficient: " <<  spring_boundary  << std::endl;
		std::cout << "Setting rigid body damping coefficient: " << damping			<< std::endl;
	}

	void SetSurfaceTensionAdhesionCoefficients(const float& surface_tension_gamma, const float& surface_adhesion_beta)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_surface_tension_gamma,	&surface_tension_gamma,	sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_const_surface_adhesion_beta,	&surface_tension_gamma, sizeof(float) ) );

		std::cout << "Setting surface tension coefficient gamma: " << surface_tension_gamma << std::endl;
		std::cout << "Setting surface adhesion coefficient beta: " << surface_adhesion_beta << std::endl;
	}

	void SetGridResolution( const unsigned int& grid_resolution_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( gridResolution, &grid_resolution_, sizeof(unsigned int) ) );
		std::cout << "Setting grid resolution radius " << grid_resolution_ << std::endl;
	}

	void SetBlockSize( const unsigned int& block_size_ )
	{
		unsigned int block_Size_cube = block_size_ * block_size_ * block_size_;
		float f = 1.0 / block_size_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( block_size, &block_size_, sizeof( unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( block_size3, &block_Size_cube, sizeof( unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invblock_size, &f, sizeof(float) ) );

		std::cout << "Setting block size " << block_size_ << std::endl;
	}

	void SetLutSize( const unsigned int& lutSize_ )
	{
		float f = 1.0 / lutSize_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutSize, &lutSize_, sizeof( unsigned int ) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invlutSize, &f, sizeof( float ) ) );
		//f  *= globalSupportRadius;
		//CUDA_SAFE_CALL( cudaMemcpyToSymbol( globalSupportRadiusByLutSize, &f, sizeof( float ) ) );
	}

	void SetKernelSelf( const float& kernelSelf_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( kernelSelf, &kernelSelf_, sizeof(float) ) );
		std::cout << "Setting KernelSelf " << kernelSelf_ << std::endl;
	}

	void SetInitialMass( const float& initialMass_ )
	{
		float f = 1.0 / initialMass_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( initialMass, &initialMass_, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invinitialMass, &f, sizeof(float) ) );
		f = initialMass_ * initialMass_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( initialMass2, &f, sizeof(float) ) );
		std::cout << "Setting initial mass " << initialMass_ << std::endl;
	}

	void SetFluidRestDensity( const float& fluidRestDensity_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidRestDensity, &fluidRestDensity_, sizeof(float) ) );
		float f = 1.0 / fluidRestDensity_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invfluidRestDensity, &f, sizeof(float) ) );
		std::cout << "Setting fluid rest density " << fluidRestDensity_ << std::endl;
	}

	void SetFluidRestVolume( const float& initialMass, const float& fluidRestDensity)
	{
		float vol = initialMass / fluidRestDensity;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidRestVolume, &vol, sizeof(float) ) );
		std::cout << "Setting fluid rest volume for PCISPH " << vol << std::endl;
	}

	void SetZindexStartingVec( const float3& zindexStartingVec_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(zindexStartingVec, &zindexStartingVec_, 3*sizeof(float) ) );
	}

	void SetGamma(const float& gamme_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( gamma, &gamme_, sizeof(float) ) );
		std::cout << "Setting gamma " << gamme_ << std::endl;
	}

	void SetFluidGasConstant( const float& fluidGasConstant_, const float& fluidGasConstantWCSPH_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidGasConstant, &fluidGasConstant_, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidGasConstantWCSPH, &fluidGasConstantWCSPH_, sizeof(float) ) );
		std::cout << "Setting fluid gas constant for standard SPH " << fluidGasConstant_ << std::endl;
		std::cout << "Setting fluid gas constant for WCSPH " << fluidGasConstantWCSPH_ << std::endl;
	}

	void SetDensityErrorFactorPCISPH( const float& denErrorFactor_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( densityErrorFactor, &denErrorFactor_, sizeof(float) ) );
		std::cout << "Setting fluid density error factor for PCISPH " << denErrorFactor_ << std::endl;
	}

	void SetBoundaryConstants( const float& forceDistance_, const float& maxBoundaryForce_, const float3& minCB_, const float3& maxCB_, bool &add_boundary_force_)
	{
		float f = 1.0 / forceDistance_; 
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( addBoundaryForce, &add_boundary_force_, sizeof(bool) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( forceDistance, &forceDistance_, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( maxBoundaryForce, &maxBoundaryForce_, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( minCollisionBox, &minCB_, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( maxCollisionBox, &maxCB_, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invforceDistance, &f, sizeof(float) ) );

		std::cout << "Setting Boundary Force State : " << ((true == add_boundary_force_)?" true ": "false") << std::endl;
		std::cout << "Setting force distance " << forceDistance_ << std::endl;
		std::cout << "Setting boundary force " << maxBoundaryForce_ << std::endl;
	}

	void SetBoundingBox( const float3& minBB_, const float3& maxBB_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( minBoundingBox, &minBB_, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( maxBoundingBox, &maxBB_, sizeof(float3) ) );
	}

	void SetRealContainerValues(const float& length, const float& height, const float& width, const float3& minB, const float3& maxB)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_box_length, &length, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_box_height, &height, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_box_width,  &width,  sizeof(float) ) );

		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_min_container_box, &minB, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_const_max_container_box, &maxB, sizeof(float3) ) );
	}

	void SetDeltaT( const float& deltaT_, const float& deltaTWCSPH_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( deltaT, &deltaT_, sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( deltaTWCSPH, &deltaTWCSPH_, sizeof(float) ) );
		std::cout << "Setting deltaT " << deltaT_ << std::endl;
		std::cout << "Setting deltaTWCSPH " << deltaTWCSPH_ << std::endl;
	}

	void SetScales( const float3& scales_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( scales, &scales_, sizeof(float3) ) );
		std::cout << "Setting scales " << scales_.x << " " << scales_.y << " " << scales_.z <<  std::endl;
	}

	void SetGravityConstant( const float& gravityConstant_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( gravityConstant, &gravityConstant_, sizeof(float) ) );
		std::cout << "Setting gravity constant " << gravityConstant_ <<  std::endl;
	}

	void SetInitialMassIntoGravityRatio( const float& initialMass_, const float& gravityConstant_ )
	{
		float f = initialMass_ * gravityConstant_;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( initialMassIntoGravityConstant, &f, sizeof(float) ) );		
	}

	void SetFluidViscosityConstant( const float& fluidViscosityConstant_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidViscosityConstant, &fluidViscosityConstant_, sizeof(float) ) );
		std::cout << "Setting main Visc " << fluidViscosityConstant_ << std::endl;
	}

	void SetFluidViscosityConstantTube(float& fluidViscosityConstant_tube_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( fluidViscosityConstant_tube, &fluidViscosityConstant_tube_, sizeof(float) ) );
		std::cout << "Setting pipe Visc " << fluidViscosityConstant_tube_ << std::endl;
	}

	void SetSpacing( const float& spacing_ )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( spacing, &spacing_, sizeof(float) ) );
		std::cout << "Setting spacing " << spacing_ << std::endl;
	}

	void SetParticleRenderingSize(const float& pradius_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( particleRenderingSize, &pradius_, sizeof(float) ) );
		std::cout << "Setting particle radius " << pradius_ << std::endl;
	}

	void SetOtherConstants()
	{
		const unsigned int XMASK_ = 0xB6DB6DB6;
		const unsigned int YMASK_ = 0x6DB6DB6D;
		const unsigned int ZMASK_ = 0xDB6DB6DB;

		CUDA_SAFE_CALL( cudaMemcpyToSymbol( xMASK, &XMASK_, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( yMASK, &YMASK_, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( zMASK, &ZMASK_, sizeof(unsigned int) ) );

		unsigned int intMax_ = std::numeric_limits<unsigned int>::max();
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( intMax, &intMax_, sizeof(unsigned int) ) );

		/*
		float3 minBBL, maxBBL;
		float3 scalesL;
		float invblock_sizeL;

		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &minBBL, minBoundingBox, sizeof(float3) ) );	
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &maxBBL, maxBoundingBox, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &scalesL, scales, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &invblock_sizeL, invblock_size, sizeof(float) ) );

		uint dimx = ceil((maxBBL.x-minBBL.x) * scalesL.x * invblock_sizeL);
		uint dimy = ceil((maxBBL.y-minBBL.y) * scalesL.y * invblock_sizeL);
		uint dimz = ceil((maxBBL.z-minBBL.z) * scalesL.z * invblock_sizeL);

		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimX, &dimx, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimY, &dimy, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimZ, &dimz, sizeof(unsigned int) ) );*/
	}

	void CreateLutKernelM4( float* A, const int& lutSize )
	{
		//CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelM4Table, A, (lutSize+1)*sizeof(float) ));
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelM4Table, A, lutSize*sizeof(float) ));
	}

	void CreateLutKernelPressureGradient( float* A, const int& lutSize )
	{
		//CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelPressureGradientTable, A, (lutSize+1)*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelPressureGradientTable, A, lutSize*sizeof(float) ) );
	}

	void CreateLutKernelViscosityLap( float* A, const int& lutSize )
	{
		//CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelViscosityLapTable, A, (lutSize+1)*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutKernelViscosityLapTable, A, lutSize*sizeof(float) ) );
	}

	void CreateLutSplineSurfaceTension( float* A, const int& lutSize )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutSplineSurfaceTensionTable, A, lutSize*sizeof(float) ) );
	}

	void CreateLutSplineSurfaceAdhesion( float* A, const int& lutSize )
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( lutSplineSurfaceAdhesionTable, A, lutSize*sizeof(float) ) );
	}	

	void SetPipePoints(const float3& p1_, const float3& p2_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( pipePoint1, &p1_, sizeof(float3) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( pipePoint2, &p2_, sizeof(float3) ) );

		float3 p;
		float l;
		p.x = p2_.x - p1_.x;
		p.y = p2_.y - p1_.y;
		p.z = p2_.z - p1_.z;
		l = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

		CUDA_SAFE_CALL( cudaMemcpyToSymbol( pipeLength, &l, sizeof(float) ) );

		float ltmp = l * l;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( pipeLength2, &ltmp, sizeof(float) ) );

		ltmp = 1.0/l;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( invpipeLength, &ltmp, sizeof(float) ) );

		std::cout << "Setting pipe points " << std::endl;
		std::cout << p1_.x << " " << p1_.y << " " << p1_.z << std::endl;
		std::cout << p2_.x << " " << p2_.y << " " << p2_.z << std::endl;
		std::cout << "Pipelength " << l << " and inverse " << 1.0/l << std::endl;
	}

	void SetPipeRadius(float& pipeRadius_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( pipeRadius, &pipeRadius_, sizeof(float) ) );
		std::cout << "Setting PipeRadius " << pipeRadius_ << std::endl;
	}

	void SetWallConstant(const float& wallX_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( wallX, &wallX_, sizeof(float) ) );
		std::cout << "Setting wall constant " << wallX_ << std::endl;
	}

	void SetGridDimensions(uint& dimX_, uint& dimY_, uint& dimZ_)
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimX, &dimX_, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimY, &dimY_, sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( dimZ, &dimZ_, sizeof(unsigned int) ) );

		uint totalBlocks = dimX_ * dimY_ * dimZ_;
		std::cout << "Setting Total Blocks : " << totalBlocks << std::endl;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( total_blocks, &totalBlocks, sizeof(unsigned int) ) );

		std::cout << "Setting grid dimensions " << dimX_ << " " << dimY_ << " " << dimZ_ << std::endl;
	}

	void SetMaximumArrayLength(uint& arrL_)
	{
		unsigned int arrL = arrL_/3.0;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( maxArrayLength, &arrL, sizeof(unsigned int) ) );
	}

	void SetZIndexTable( unsigned short* array_zindex )
	{
		//CUDA_SAFE_CALL( cudaMemcpyToSymbol( &zIndexTable[0][0], array_zindex, 1024*3*sizeof(unsigned short) ) );
	}
}

//========================================================================
//								KERNEL FUNCTIONS
//========================================================================

__global__ 
	void CopySortedParticleValuesKernel( unsigned int particleCount, 
										 float4* array_pos_, 
										 float4* array_vel_, 
										 int* array_type_, 
										 unsigned int* particlesKVPair_d )
{	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < PCOUNT )
	{
		unsigned int value = particlesKVPair_d[2*index+1];
		//uint value = index;
		array_pos_[index]			= tex1Dfetch(texture_pos_zindex, value);
		array_vel_[index]			= tex1Dfetch(texture_vel, value);
		array_type_[index]			= tex1Dfetch(texture_type, value);
		//particlesKVPair_d[2*index+1] = index;
	}
}

__global__ 
	void CopySortedParticleValuesKernelApproximate( unsigned int particleCount, 
	float4* array_pos_, 										
	float4* array_vel_, 										
	int* array_active_type_, 
	unsigned int* particlesKVPair_d )
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < PCOUNT )
	{
		unsigned int value			= particlesKVPair_d[2*index+1];
		array_pos_[index]			= tex1Dfetch(texture_pos_zindex, value);
		array_vel_[index]			= tex1Dfetch(texture_vel, value);
		array_active_type_[index]	= tex1Dfetch(texture_active_type, value);
	}
}

__global__ 
	void CopySortedParticleValuesKernelRigidBodySPH( unsigned int particleCount, 
	float4* array_pos_, 
	float4* array_vel_, 
	float4* array_relative_pos_,
	int* array_type_, 
	int* array_parent_rb,
	int* array_order_in_child_particles_array,
	unsigned int* particlesKVPair_d )
{	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < PCOUNT )
	{
		unsigned int value = particlesKVPair_d[2*index+1];
		//uint value = index;
		array_pos_[index]			= tex1Dfetch(texture_pos_zindex, value);
		array_relative_pos_[index]  = tex1Dfetch(texture_relative_pos, value);
		array_vel_[index]			= tex1Dfetch(texture_vel, value);
		array_type_[index]			= tex1Dfetch(texture_type, value);
		array_parent_rb[index]		= tex1Dfetch(texture_parent_rb, value);
		array_order_in_child_particles_array[index] = tex1Dfetch(texture_order_in_child_particles_array, value);
		//particlesKVPair_d[2*index+1] = index;
	}
}

__global__ 
	void CopySortedParticleValuesKernelStaticBoundariesPCISPH(unsigned int particleCount, 
	float4* array_pos_, 
	float4* array_vel_, 
	int* array_type_, 
	float* array_previous_corr_pressure_,
	float* array_previous_predicted_density_,
	unsigned int* particlesKVPair_d)
{	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < PCOUNT )
	{
		unsigned int value = particlesKVPair_d[2*index+1];
		//uint value = index;
		array_pos_[index]								= tex1Dfetch(texture_pos_zindex, value);
		array_vel_[index]								= tex1Dfetch(texture_vel, value);
		array_type_[index]								= tex1Dfetch(texture_type, value);
		array_previous_corr_pressure_[index]			= tex1Dfetch(texture_corr_pressure, value);
		array_previous_predicted_density_[index]		= tex1Dfetch(texture_predicted_density, value);
		//particlesKVPair_d[2*index+1] = index;
	}
}

__global__ 
	void CopySortedParticleValuesKernelRigidBodyPCISPH(unsigned int particleCount, 
													   float4* array_pos_, 
													   float4* array_vel_, 
													   float4* array_relative_pos_,
													   int* array_type_, 
													   int* array_parent_rb_,
													   int* array_order_in_child_particles_array_,
													   float* array_previous_corr_pressure_,
													   float* array_previous_predicted_density_,
													   unsigned int* particlesKVPair_d)
{	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < PCOUNT )
	{
		unsigned int value = particlesKVPair_d[2*index+1];
		//uint value = index;
		array_pos_[index]								= tex1Dfetch(texture_pos_zindex, value);
		array_relative_pos_[index]  					= tex1Dfetch(texture_relative_pos, value);
		array_vel_[index]								= tex1Dfetch(texture_vel, value);
		array_type_[index]								= tex1Dfetch(texture_type, value);
		array_parent_rb_[index]							= tex1Dfetch(texture_parent_rb, value);
		array_order_in_child_particles_array_[index]	= tex1Dfetch(texture_order_in_child_particles_array, value);
		array_previous_corr_pressure_[index]			= tex1Dfetch(texture_corr_pressure, value);
		array_previous_predicted_density_[index]		= tex1Dfetch(texture_predicted_density, value);
		//particlesKVPair_d[2*index+1] = index;
	}
}

__global__
	void ClearIndexArrays(uint* indices)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < 2*total_blocks )
	{
		if(index % 2 == 0)
			indices[index] = PCOUNT;		// grid_start_index: init to PCOUNT, reason: see atomicMin((unsigned int*)&grid_ancillary_array[2*current_block_index], index );
		else
			indices[index] = 0;				// num_particles: init to 0, reason: see atomicInc((unsigned int*)&grid_ancillary_array[2*current_block_index+1], PCOUNT);
	}
}

__global__
	void CalculateBlocksKernel(uint* grid_ancillary_array, uint* indices2, uint* tmpArray)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	int current_block_index;			 
	int compared_particle_block_index;
	int startIndex;

	if( index < PCOUNT )
	{
		float4 p = tex1Dfetch( texture_pos_zindex, index );
		float4 q = make_float4( maxBoundingBox.x, maxBoundingBox.y, maxBoundingBox.z, intMax );				// init q for index == 0
		if( index > 0 )
			q = tex1Dfetch( texture_pos_zindex, index-1 );

		int x = p.x * scales.x * invblock_size;
		int y = p.y * scales.y * invblock_size;
		int z = p.z * scales.z * invblock_size;
		//int current_block_index = (dimX * dimZ) * y + (x * dimZ + z);
		current_block_index = (dimX * dimY) * z + (y * dimX + x);


		int x1 = q.x * scales.x * invblock_size;
		int y1 = q.y * scales.y * invblock_size;
		int z1 = q.z * scales.z * invblock_size;
		//int compared_particle_block_index = (dimX * dimZ) * y1 + (x1 * dimZ + z1);
		compared_particle_block_index = (dimX * dimY) * z1 + (y1 * dimX + x1);

		// Starting index of particles within this block
		atomicMin((unsigned int*)&grid_ancillary_array[2*current_block_index], index );
		// Count of particles in this block
		atomicInc((unsigned int*)&grid_ancillary_array[2*current_block_index+1], PCOUNT);
		//atomicMax((unsigned int*)&grid_ancillary_array[2*current_block_index+1], index);

		// add new block cells if we find every first particle in each block 
		if( current_block_index != compared_particle_block_index )
		{
			int newBlock = atomicInc( (unsigned int*)&blockCount, total_blocks );
			indices2[3*newBlock] = current_block_index;		// indices2 has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		}
	}

	/*__syncthreads();

	if( index < PCOUNT )
	{
	startIndex = grid_ancillary_array[2*current_block_index];

	// If different blocks, then start a new block
	if( current_block_index != compared_particle_block_index && startIndex == index )
	{
	int newBlock = atomicInc( (unsigned int*)&blockCount, total_blocks );
	indices2[3*newBlock] = dest;
	}
	}*/
}

// If some block contains more particles than MAX_THREADS_PER_BLOCK
// a new block is created in indices2 in it
__global__
	void RedistributeBlocksKernel2(uint* grid_ancillary_array, uint* indices2)
{
	if( threadIdx.x == 0 )
	{
		uint bid = indices2[3*blockIdx.x];

		uint startIndex = grid_ancillary_array[2*bid];
		uint length = grid_ancillary_array[2*bid+1];

		indices2[3*blockIdx.x+1] = startIndex;
		indices2[3*blockIdx.x+2] = length;

		uint startIndex2 = startIndex;
		int length2 = length;
		int quanta2;

		if( length > MAX_THREADS_PER_BLOCK_SPH )
		{
			indices2[3*blockIdx.x+2] = MAX_THREADS_PER_BLOCK_SPH;

			startIndex2 += MAX_THREADS_PER_BLOCK_SPH;
			length2 -= MAX_THREADS_PER_BLOCK_SPH;

			while( startIndex2 < startIndex + length )
			{
				quanta2 = MAX_THREADS_PER_BLOCK_SPH;
				if(length2 < MAX_THREADS_PER_BLOCK_SPH)
					quanta2 = length2;

				int newBlock = atomicInc( (unsigned int*)&blockCount, maxArrayLength );
				indices2[3*newBlock] = bid;					// currBlockIndex  
				indices2[3*newBlock+1] = startIndex2;		// currStartIndex
				indices2[3*newBlock+2] = quanta2;			// currNumParticles

				startIndex2 += quanta2;
				length2 -= quanta2;
			}
		}
	}
}

__global__
	void RedistributeBlocksKernel(uint* grid_ancillary_array, uint* indices2)
{
	if( threadIdx.x == 0 )
	{
		uint currBlockIndex = indices2[3*blockIdx.x];
		uint count = grid_ancillary_array[2*currBlockIndex+1];
		uint startIndex = grid_ancillary_array[2*currBlockIndex];

		indices2[3*blockIdx.x+1] = startIndex;
		indices2[3*blockIdx.x+2] = count;


		if( count > MAX_THREADS_PER_BLOCK_SPH )
		{
			indices2[3*blockIdx.x+2] = MAX_THREADS_PER_BLOCK_SPH;
			count -= MAX_THREADS_PER_BLOCK_SPH;

			uint loopcount = MAX_THREADS_PER_BLOCK_SPH;
			if( count < MAX_THREADS_PER_BLOCK_SPH )
				loopcount = count;
			uint shiftStart = MAX_THREADS_PER_BLOCK_SPH;
			while( count > 0 )
			{
				//int newBlock = atomicInc( (unsigned int*)&grid_ancillary_array[2*endL], PCOUNT );
				int newBlock = atomicInc( (unsigned int*)&blockCount, maxArrayLength );
				indices2[3*newBlock] = currBlockIndex;					// currBlockIndex
				indices2[3*newBlock+1] = startIndex + shiftStart;		// currStartIndex
				indices2[3*newBlock+2] = loopcount;						// currNumParticles

				count -= loopcount;
				shiftStart += loopcount;
				loopcount = MAX_THREADS_PER_BLOCK_SPH;
				if( count <= MAX_THREADS_PER_BLOCK_SPH )
					loopcount = count;
			}
		}
	}
}

__global__
	void CalculateDensitiesInBlocksKernelSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float density = initialMass * kernelSelf;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		//smoothedColor = density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new densities from new copied neighbors in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
					distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

					//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					density += (distance * initialMass );
					//smoothedColor += initialMass * distance * neighInvDensities[j];
					//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		array_pressure_[index] = fmaxf( 0.0f, (density * invfluidRestDensity - 1.0f) * fluidRestDensity * fluidGasConstant );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}


__global__
	void CalculateDensitiesInBlocksKernelSPHApproximate(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int activeState = tex1Dfetch(texture_active_type, index);
	if (activeState == 0 || activeState == 1)
	{
		__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ float neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

		__shared__ uint someMaxValue;

		float density = initialMass * kernelSelf;
		//float smoothedColor;
		//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

		uint neighStartIndex = 0;
		uint neighNumParticles = 0;
		//int offset = 0;
		int i = 0, j = 0, k = 0;
		int ii = 0, jj = 0, kk = 0;

		int N = currBlockIndex;

		uint tmpvari = dimX * dimY;
		while( N >= tmpvari )
		{
			N -= tmpvari;
			k++;
		}

		while( N >= dimX )
		{
			N -= dimX;
			j++;
		}

		i = N;

		/*tmpvari = dimX * dimZ;
		while( N >= tmpvari )
		{
		N -= tmpvari;
		j++;
		}

		while( N >= dimZ )
		{
		N -= dimZ;
		i++;
		}

		k = N;*/

		N = tid;
		if( N < 27 )
		{
			while( N >= 9 )
			{
				N -= 9;
				kk++;
			}
			kk--;

			while( N >= 3 )
			{
				N -= 3;
				jj++;
			}
			jj--;

			ii = N - 1;

			i = i + ii;
			j = j + jj;
			k = k + kk;

			//N = (dimX * dimZ) * j + (dimZ * i + k);
			//N = (dimX * dimY) * k + (dimY * i + j);
			N = (dimX * dimY) * k + (dimX * j + i);

			neighStartIndex = 0;
			neighNumParticles = 0;
			if( N >= 0 && N < dimX * dimY * dimZ )
			{
				neighStartIndex = grid_ancillary_array[2*N];
				neighNumParticles = grid_ancillary_array[2*N+1];
			}
		}
		particleCountBlocks[tid] = neighNumParticles;

		__syncthreads();

		float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
		if( tid < currNumParticles )
		{
			p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
			//smoothedColor = density / tex1Dfetch( texture_density, index );
		}

		// Determine M: max particles in neighbors
		if( tid == 0 )
		{
			someMaxValue = 0;
			for( i = 0; i < 27; i++ )
			{
				someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
			}
		}	

		__syncthreads();

		for( i = 0; i < someMaxValue; i++ )
		{
			neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
			neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
			neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
			neigh_pos_zindex[tid].w = intMax;

			// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
			if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
			{
				neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
				//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			}

			__syncthreads();

			// Accumulate new densities from new copied neighbors in shared memory
			if( tid < currNumParticles )
			{
				for( j = 0; j < 27; j++ )
				{
					float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

					if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
					{
						tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
						distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

						//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
						//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

						density += (distance * initialMass );
						//smoothedColor += initialMass * distance * neighInvDensities[j];
						//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
					}
				}
			}

			__syncthreads();
		}

		//__syncthreads();

		// Write updated densities to global memory
		if( tid < currNumParticles && index < PCOUNT )
		{
			array_density[index] = density;
			array_pressure_[index] = fmaxf( 0.0f, (density * invfluidRestDensity - 1.0f) * fluidRestDensity * fluidGasConstant );
			//array_smoothcolor_normal[index].w = smoothedColor;
			//array_smoothcolor_normal[index] = normal;	
		}
	}
}

__global__
	void FindActiveParticlesInBlocksKernelSPH(int* array_particle_state_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 sumDistWeightedMass = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  sumMassOfNeighbors = 0.0f;
	
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
		}

		__syncthreads();

		// calculate gradient color field for each particle using its neighbor's information
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					sumMassOfNeighbors  += initialMass;
					sumDistWeightedMass += (p_pos_zindex - neigh_pos_zindex[j]) * initialMass;
				}
			}
			sumDistWeightedMass *= (1.0f/sumMassOfNeighbors);
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		if( getLength(sumDistWeightedMass) >= d_const_dist_to_center_mass_cutoff )
		{
			array_particle_state_[index] = 0;
		}
	}
}

__global__
	void MarkMoreActiveParticlesStateInBlocksKernelSPH(int* array_particle_state_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel		= make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel		 = tex1Dfetch( texture_vel, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
		}

		__syncthreads();

		// calculate gradient color field for each particle using its neighbor's information
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					if (getLength(p_vel) >= d_const_vel_cutoff)
					{
						// mark all particles in N_i(t) active
						atomicExch(&array_particle_state_[neighStartIndex + i], 0);
					}
					else if (array_particle_state_[neighStartIndex + i] != 0)
					{
						// mark all non-active neighbors semi-active
						atomicExch(&array_particle_state_[neighStartIndex + i], 1);
					}
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();
}

__global__
	void MarkAllRemainingParticlesStateInBlocksKernelSPH(int* array_particle_state_, uint* indices1, uint* indices2 )
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;

	if( index < PCOUNT )
	{
		int activeState = array_particle_state_[index];
		if (activeState != 0 && activeState != 1)
		{
			array_particle_state_[index] = 2;
		}		
	}
}

__global__
	void CalculateDensitiesInBlocksKernelWCSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float density = initialMass * kernelSelf;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		//smoothedColor = density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new densities from new copied neighbors in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
					distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

					//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					density += (distance * initialMass );
					//smoothedColor += initialMass * distance * neighInvDensities[j];
					//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	const float b_i = fluidRestDensity * fluidGasConstantWCSPH / gamma;
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		const float powGamma = powf(density * invfluidRestDensity, gamma);
		array_pressure_[index] = fmaxf( 0.0f, (powGamma - 1.0f) * b_i );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void CalculateDensitiesInBlocksKernelWCSPHApproximate(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float density = initialMass * kernelSelf;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		//smoothedColor = density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new densities from new copied neighbors in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
					distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

					//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					density += (distance * initialMass );
					//smoothedColor += initialMass * distance * neighInvDensities[j];
					//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	const float b_i = fluidRestDensity * fluidGasConstantWCSPH / gamma;
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		const float powGamma = powf(density * invfluidRestDensity, gamma);
		array_pressure_[index] = fmaxf( 0.0f, (powGamma - 1.0f) * b_i );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void CalculateForcesInBlocksKernelSPH( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure = 0.0f;
	float4 p_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		p_pressure = tex1Dfetch( texture_pressure, index);
		p_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
		invdensity = 1.0  / tex1Dfetch( texture_density, index );
		pVol = initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;
					float tmpvarf = lutKernelPressureGradientTable[dist_lut];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					tmpvarf = lutKernelViscosityLapTable[dist_lut];

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

					p_force -= ((p_vel - neigh_vel[j]) * tmpVis *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#else
					p_force -= ((p_vel - neigh_vel[j]) * fluidViscosityConstant *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#endif


					tmpvarf = p_pressure * invdensity * invdensity + 
						neigh_pressure[j] * neigh_invdensities[j] * neigh_invdensities[j];
					p_force -= kernelGradient * tmpvarf * initialMass2;	

					if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
					{
						p_force -= CalculateSurfaceTensionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j]);
					}
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		// Handle boundary forces
		p_force.y -= initialMassIntoGravityConstant;
		p_force += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

		// Update particle positions & velocities
		p_vel += (p_force * invinitialMass * deltaT );
		p_pos_zindex += (p_vel * deltaT);

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
		CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

		//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
	}
	__syncthreads();

	// Write updated positions to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		array_pos_[index] = p_pos_zindex;
		array_vel_[index] = p_vel;
		//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
	}
}

__global__
	void CalculateForcesInBlocksKernelSPHApproximate( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int activeType = tex1Dfetch(texture_active_type, index);
	if (activeType == 0)
	{
		__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ float neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
		__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
		//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

		uint neighStartIndex = 0;
		uint neighNumParticles = 0;
		int i = 0, j = 0, k = 0;
		int ii = 0, jj = 0, kk = 0;

		float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
		float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
		float  p_pressure = 0.0f;
		float4 p_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
		//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
		float invdensity;
		float pVol;

		int N = currBlockIndex;

		uint tmpvari = dimX * dimY;
		while( N >= tmpvari )
		{
			N -= tmpvari;
			k++;
		}

		while( N >= dimX )
		{
			N -= dimX;
			j++;
		}

		i = N;

		/*tmpvari = dimX * dimZ;
		while( N >= tmpvari )
		{
		N -= tmpvari;
		j++;
		}

		while( N >= dimZ )
		{
		N -= dimZ;
		i++;
		}

		k = N;*/

		N = tid;
		if( tid < 27 )
		{
			while( N >= 9 )
			{
				N -= 9;
				kk++;
			}
			kk--;

			while( N >= 3 )
			{
				N -= 3;
				jj++;
			}
			jj--;

			ii = N - 1;

			i = i + ii;
			j = j + jj;
			k = k + kk;

			//N = (dimX * dimZ) * j + (dimZ * i + k);
			//N = (dimX * dimY) * k + (dimY * i + j);
			N = (dimX * dimY) * k + (dimX * j + i);

			neighStartIndex = 0;
			neighNumParticles = 0;
			if( N >= 0 && N < dimX * dimY * dimZ )
			{
				neighStartIndex = indices1[2*N];
				neighNumParticles = indices1[2*N+1];
			}
			particleCountBlocks[tid] = neighNumParticles;
		}

		__syncthreads();

		if( tid < currNumParticles )
		{
			p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
			p_vel = tex1Dfetch( texture_vel, index );
			p_pressure = tex1Dfetch( texture_pressure, index);
			p_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
			invdensity = 1.0  / tex1Dfetch( texture_density, index );
			pVol = initialMass * invdensity;
		}

		// Determine M: max particles in neighbors
		if( tid == 0 )
		{
			someMaxValue = 0;
			for( i = 0; i < 27; i++ )
			{
				someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
			}
		}	

		__syncthreads();

		for( i = 0; i < someMaxValue; i++ )
		{
			neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
			neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
			neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
			neigh_pos_zindex[tid].w = intMax;
			neigh_vel[tid].x = 0.0;
			neigh_vel[tid].y = 0.0;
			neigh_vel[tid].z = 0.0;
			neigh_vel[tid].w = 0.0;
			neigh_pressure[tid] = 0.0;
			neigh_invdensities[tid] = 0.0;

			// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
			if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
			{
				neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
				neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
				neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
				neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
				//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
			}

			__syncthreads();

			// Compute new forces using texture densities and neighbors copied in shared memory
			if( tid < currNumParticles )
			{
				for( j = 0; j < 27; j++ )
				{
					float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

					if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
					{
						tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius;
						if( tmpvari > lutSize )
							tmpvari = lutSize;
						float tmpvarf = lutKernelPressureGradientTable[tmpvari];
						float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

						//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


						tmpvarf = lutKernelViscosityLapTable[tmpvari];

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

						p_force -= ((p_vel - neigh_vel[j]) * tmpVis *
							pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#else
						p_force -= ((p_vel - neigh_vel[j]) * fluidViscosityConstant *
							pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#endif


						tmpvarf = p_pressure * invdensity * invdensity + 
							neigh_pressure[j] * neigh_invdensities[j] * neigh_invdensities[j];
						p_force -= kernelGradient * tmpvarf * initialMass2;	
					}
				}

				//ii += bdim;
			}

			__syncthreads();
		}

		__syncthreads();

		if( tid < currNumParticles )
		{
			// Handle boundary forces
			p_force.y -= initialMassIntoGravityConstant;
			p_force += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

			// Update particle positions & velocities
			p_vel += (p_force * invinitialMass * deltaT );
			p_pos_zindex += (p_vel * deltaT);

			// Handle collisions
			CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
			CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

			//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
		}
		__syncthreads();

		// Write updated positions to global memory
		if( tid < currNumParticles && index < PCOUNT )
		{		
			array_pos_[index] = p_pos_zindex;
			array_vel_[index] = p_vel;
			//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
		}
	}
}

__global__
	void CalculateForcesInBlocksKernelWCSPH( float4* array_pos_, float4* array_vel_, 
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bdim = blockDim.x;
	const unsigned int bid = blockIdx.x;

	unsigned int index = bid * bdim + tid;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure = 0.0f;
	float4 pforce_density = make_float4( 0.0, 0.0, 0.0, 0.0 );
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		p_pressure = tex1Dfetch( texture_pressure, index);
		pforce_density = make_float4( 0.0, 0.0, 0.0, tex1Dfetch( texture_density, index ) );
		invdensity = 1.0  / pforce_density.w;
		pVol = initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;
					float tmpvarf = lutKernelPressureGradientTable[dist_lut];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					tmpvarf = lutKernelViscosityLapTable[dist_lut];

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

					pforce_density -= ((p_vel - neigh_vel[j]) * tmpVis *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#else
					pforce_density -= ((p_vel - neigh_vel[j]) * fluidViscosityConstant *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#endif


					tmpvarf = p_pressure * invdensity * invdensity + 
						neigh_pressure[j] * neigh_invdensities[j] * neigh_invdensities[j];
					pforce_density -= kernelGradient * tmpvarf * initialMass2;	

					if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
					{
						pforce_density -= CalculateSurfaceTensionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j]);
					}
				}
			}

			ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		// Handle boundary forces
		pforce_density.y -= initialMassIntoGravityConstant;
		pforce_density += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

		// Update particle positions & velocities
		p_vel += (pforce_density * invinitialMass * deltaTWCSPH );
		p_pos_zindex += (p_vel * deltaTWCSPH);

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
		CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

		//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
	}
	__syncthreads();

	// Write updated positions to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		array_pos_[index] = p_pos_zindex;
		array_vel_[index] = p_vel;
		//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
	}
}

__global__
	void CalculateForcesInBlocksKernelWCSPHApproximate( float4* array_pos_, float4* array_vel_, 
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bdim = blockDim.x;
	const unsigned int bid = blockIdx.x;

	unsigned int index = bid * bdim + tid;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure = 0.0f;
	float4 pforce_density = make_float4( 0.0, 0.0, 0.0, 0.0 );
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		p_pressure = tex1Dfetch( texture_pressure, index);
		pforce_density = make_float4( 0.0, 0.0, 0.0, tex1Dfetch( texture_density, index ) );
		invdensity = 1.0  / pforce_density.w;
		pVol = initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius;
					if( tmpvari > lutSize )
						tmpvari = lutSize;
					float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					tmpvarf = lutKernelViscosityLapTable[tmpvari];

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

					pforce_density -= ((p_vel - neigh_vel[j]) * tmpVis *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#else
					pforce_density -= ((p_vel - neigh_vel[j]) * fluidViscosityConstant *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#endif


					tmpvarf = p_pressure * invdensity * invdensity + 
						neigh_pressure[j] * neigh_invdensities[j] * neigh_invdensities[j];
					pforce_density -= kernelGradient * tmpvarf * initialMass2;	
				}
			}

			ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		// Handle boundary forces
		pforce_density.y -= initialMassIntoGravityConstant;
		pforce_density += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

		// Update particle positions & velocities
		p_vel += (pforce_density * invinitialMass * deltaTWCSPH );
		p_pos_zindex += (p_vel * deltaTWCSPH);

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
		CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

		//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
	}
	__syncthreads();

	// Write updated positions to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		array_pos_[index] = p_pos_zindex;
		array_vel_[index] = p_vel;
		//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
	}
}

__global__
	void ExtractSurfaceParticlesKernel2( float4 *array_filtered_pos_zindex, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bdim = blockDim.x;
	const unsigned int bid = blockIdx.x;

	uint newtid;
	uint tid2;

	unsigned int index = bid * bdim + tid;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 filtered_neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];	


	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	__shared__ uint lfCount;
	__shared__ uint oldLfCount;


	uint neighStartIndex;
	uint neighNumParticles;
	uint tmpvari;
	float nNeighbors = 0.0;
	int N;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 center = make_float4( 0.0, 0.0, 0.0, 0.0 );

	if( tid == 0 )
	{	
		currBlockIndex = indices2[3*bid];
		//currStartIndex = indices2[3*bid+1];
		//currNumParticles = indices2[3*bid+2];

		//currBlockIndex = bid;
		currStartIndex = indices1[2*currBlockIndex];
		currNumParticles = indices1[2*currBlockIndex+1];
		lfCount = 0;
	}


	__syncthreads();

	index = currStartIndex + tid;

	N = currBlockIndex;

	tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/


	if( tid < 27 )
	{
		N = tid;
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;


		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		N = (dimX * dimY) * k + (dimX * j + i);
		//N = (dimX * dimY) * k + (dimY * i + j);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();


	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}

	N = ceilf((float)currNumParticles/MAX_THREADS_PER_BLOCK_SPH);
	jj = 0;
	index = currStartIndex + tid;
	tid2 = tid;

	__syncthreads();

	while( jj < N )
	{
		newtid = tid2 % MAX_THREADS_PER_BLOCK_SPH;
		//newtid = WRAP( tid2, MAX_THREADS_PER_BLOCK_SPH );

		if( tid2 < currNumParticles )
		{
			p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		}

		__syncthreads();

		for( i = 0; i < someMaxValue; i++ )
		{
			neigh_pos_zindex[newtid].x = 4*maxBoundingBox.x;
			neigh_pos_zindex[newtid].y = 4*maxBoundingBox.y;
			neigh_pos_zindex[newtid].z = 4*maxBoundingBox.z;
			neigh_pos_zindex[newtid].w = intMax;

			if( i < neighNumParticles )
			{
				neigh_pos_zindex[newtid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			}

			__syncthreads();

			if( tid2 < currNumParticles )
			{
				for( j = 0; j < 27; j++ )
				{
					float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

					if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
					{
						center.x += neigh_pos_zindex[j].x;
						center.y += neigh_pos_zindex[j].y;
						center.z += neigh_pos_zindex[j].z;
						nNeighbors += 1.0;
					}
				}

				ii += bdim;
			}

			__syncthreads();
		}

		if(nNeighbors < 1.0)
			nNeighbors = 1.0;
		center.x /= nNeighbors;
		center.y /= nNeighbors;
		center.z /= nNeighbors;
		__syncthreads();

		bool isSurface = false;
		if( tid2 < currNumParticles )
		{
			if( distanceSqrt( p_pos_zindex, center ) > CENTER_OF_MASS_THRESHOLD )
				isSurface = true;
			else
				if ( nNeighbors < N_NEIGHBORS_THRESHOLD )
					isSurface = true;
		}

		/*if(isSurface)
		{
		uint oldFcount = atomicInc(&filteredCount, PCOUNT);
		array_filtered_pos_zindex[oldFcount] = p_pos_zindex;
		}*/

		for ( i = 0; i < currNumParticles; ++i )
		{
			__syncthreads();
			if ( tid == i && isSurface )
			{
				filtered_neigh_pos_zindex[lfCount] = p_pos_zindex;
				++lfCount;
			}
		}
		__syncthreads();

		/*if ( isSurface )
		{
		uint oLfCount = atomicInc( &lfCount, PCOUNT );
		filtered_neigh_pos_zindex[oLfCount] = p_pos_zindex;
		}
		__syncthreads();*/

		if  ( tid == 0 )
			oldLfCount = atomicAdd(&filteredCount, lfCount ); 
		__syncthreads();

		if( tid < lfCount )
			array_filtered_pos_zindex[oldLfCount + tid] = filtered_neigh_pos_zindex[tid];

		jj++;
		index += MAX_THREADS_PER_BLOCK_SPH;
		tid2 += MAX_THREADS_PER_BLOCK_SPH;

		if(tid == 0)
			lfCount = 0;

		__syncthreads();
	}

	__syncthreads();
}

__global__
	void ExtractSurfaceParticlesKernel( float4 *array_filtered_pos_zindex, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bdim = blockDim.x;
	const unsigned int bid = blockIdx.x;

	unsigned int index = bid * bdim + tid;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 filtered_neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];	


	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	__shared__ uint lfCount;
	__shared__ uint oldLfCount;

	uint neighStartIndex;
	uint neighNumParticles;
	uint tmpvari;
	float nNeighbors = 0.0;
	int N;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 center = make_float4( 0.0, 0.0, 0.0, 0.0 );

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
		lfCount = 0;
	}


	__syncthreads();

	index = currStartIndex + tid;

	N = currBlockIndex;
	tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	if( tid < 27 )
	{
		N = tid;
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;


		i = i + ii;
		j = j + jj;
		k = k + kk;

		N = (dimX * dimY) * k + (dimY * i + j);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];			
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();


	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;

		if( i < neighNumParticles )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
		}

		__syncthreads();

		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					center.x += neigh_pos_zindex[j].x;
					center.y += neigh_pos_zindex[j].y;
					center.z += neigh_pos_zindex[j].z;
					nNeighbors += 1.0;
				}
			}

			ii += bdim;
		}

		__syncthreads();
	}

	center.x /= nNeighbors;
	center.y /= nNeighbors;
	center.z /= nNeighbors;

	bool isSurface = false;
	if( tid < currNumParticles && index < PCOUNT )
	{
		if( distanceSqrt( p_pos_zindex, center ) > CENTER_OF_MASS_THRESHOLD )
			isSurface = true;
		else
			if ( nNeighbors < N_NEIGHBORS_THRESHOLD )
				isSurface = true;
	}

	/*if(isSurface)
	{
	uint oldFcount = atomicInc(&filteredCount, PCOUNT);
	array_filtered_pos_zindex[oldFcount] = p_pos_zindex;
	}*/

	for ( i = 0; i < currNumParticles; ++i )
	{
		__syncthreads();
		if ( tid == i && isSurface )
		{
			filtered_neigh_pos_zindex[lfCount] = p_pos_zindex;
			++lfCount;
		}
	}
	__syncthreads();

	/*if ( isSurface )
	{
	uint oLfCount = atomicInc( &lfCount, PCOUNT );
	filtered_neigh_pos_zindex[oLfCount] = p_pos_zindex;
	}*/
	__syncthreads();

	if  ( tid == 0 )
		oldLfCount = atomicAdd( &filteredCount, lfCount ); 
	__syncthreads();

	if( tid < lfCount )
		array_filtered_pos_zindex[oldLfCount + tid] = filtered_neigh_pos_zindex[tid];
}

__global__
	void FillZindicesKernel(const float4* array_pos_, uint* indices)
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;
	float4 pos;
	float4 startVec = make_float4(zindexStartingVec.x, zindexStartingVec.y, zindexStartingVec.z, 0.0f);	// TODO: store zindexStartingVec in constant memory 

	if( index < PCOUNT )
	{
		//pos = array_pos_[index] - startVec;	// TODO: this doesn't work
		pos = array_pos_[index];

		int x = pos.x * scales.x * invblock_size;
		int y = pos.y * scales.y * invblock_size;
		int z = pos.z * scales.z * invblock_size;
		int key_zindex = (dimX * dimY) * z + (y * dimX + x);		// TODO: Why? It looks like just a grid cell index, how could this be interpreted as zindex of a particle???

		indices[2*index] = key_zindex;
		indices[2*index+1] = index;
	}
}

__global__ void GetReductionFinalMax(float* idata, int numPnts, float* max_predicted_density)
{
	uint tid = threadIdx.x;
	if (tid == 0)
	{
		float maxValue = 0.0f;
		for (int i = 0; i < numPnts; ++i)
		{
			if(idata[i] > maxValue)
				maxValue = idata[tid];		
		}
		*max_predicted_density = maxValue;
	}
}

__global__
	void CalculateExternalForcesInBlocksKernelPCISPH( float4* static_external_force_, float4* corr_pressure_force_, float* correction_pressure_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 pforce_ = make_float4( 0.0, 0.0, 0.0, 0.0 );
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	// Add viscosity force
	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel_pressure[tid].x = 0.0;
		neigh_vel_pressure[tid].y = 0.0;
		neigh_vel_pressure[tid].z = 0.0;
		neigh_vel_pressure[tid].w = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel_pressure[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute external forces using texture attributes and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					tmpvari = dist * lutSize * invglobalSupportRadius;
					if( tmpvari > lutSize )
						tmpvari = lutSize;

					//normal -= kernelGradient * neigh_smoothedColor[j] * fluidRestVolume;

					float tmpvarf = lutKernelViscosityLapTable[tmpvari];

					// (1) simple viscosity force model
					pforce_ -= ((p_vel - neigh_vel_pressure[j]) * fluidViscosityConstant * fluidRestVolume * tmpvarf * fluidRestVolume);

					// (2) viscosity force according to E.q(10) from paper "Weakly compressible SPH for free surface flows"
					/*
					float3 X_ab = make_float3(neigh_pos_zindex[j].x - p_pos_zindex.x, neigh_pos_zindex[j].y - p_pos_zindex.y, neigh_pos_zindex[j].z - p_pos_zindex.z); // X_ab = X_b - X_a
					float3 V_ab = make_float3(neigh_vel_pressure[j].x - p_vel.x, neigh_vel_pressure[j].y - p_vel.y, neigh_vel_pressure[j].z - p_vel.z); // V_ab = V_b - V_a;
					float temp = getDotProduct(X_ab, V_ab);
					if (temp < 0)
					{
						float alpha = 0.002;	// alpha is in between 0.08 and 0.5 Need to be tunes for our own purpose
						float C_s = 3.13; // Using the estimation v_f = sqrt(2*g*H) for h = 0.5m, 
						float v = (alpha * d_const_particle_radius * C_s) * invfluidRestDensity;	// rho_a = rho_b = rest density in our incompressible fluid case
						float epsilon = 0.01; 
						float PI_ab = -v * temp / ( getDotProduct(X_ab, X_ab) + epsilon * d_const_particle_radius * d_const_particle_radius);
						const float4 x_ij = make_float4(p_pos_zindex.x-neigh_pos_zindex[j].x, p_pos_zindex.y-neigh_pos_zindex[j].y, p_pos_zindex.z-neigh_pos_zindex[j].z, p_pos_zindex.w-neigh_pos_zindex[j].w);
						pforce_ -= x_ij * tmpvarf * initialMass * initialMass * PI_ab;
					}
					*/

					// surface tension force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
					tmpvarf = lutSplineSurfaceTensionTable[tmpvari];
					if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
					{
						pforce_ -= (p_pos_zindex - neigh_pos_zindex[j]) * d_const_surface_tension_gamma * initialMass * initialMass * tmpvarf * (1.0f/dist);
					}				
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		// Add gravity force	
		pforce_.y -= initialMassIntoGravityConstant;

		// we avoid using addBoundaryForce in here for performance reason 
		// Add boundary forces
		pforce_ += AddBoundaryForcePCISPH( p_pos_zindex);

		// Add other external forces in here


	}
	__syncthreads();

	// Write updated forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		static_external_force_[index]  = pforce_;
		corr_pressure_force_[index]    = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // init some quantities which are going to be used in the prediction step
		correction_pressure_[index]    = 0.0f;
	}
}

__global__
	void PredictPositionAndVelocityBlocksKernelPCISPH( float4* predicted_pos_)
{
	// v' = v + delta_t * a
	// a = F / m
	// v' = v + delta_t * F * V / m
	// v' = v + delta_t * F * 1/density

	// compute predicted position and velocity
	// this method is called when using PCISPH
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 external_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 corr_pressure_force = make_float4( 0.0, 0.0, 0.0, 0.0 );

	if( index < PCOUNT )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		external_force = tex1Dfetch( texture_static_force, index);
		corr_pressure_force = tex1Dfetch( texture_corr_pressure_force, index);

		// compute predicted position and velocity
		p_vel += ( ( external_force + corr_pressure_force ) * invinitialMass * deltaT );		// predicted velocity
		p_pos_zindex += (p_vel * deltaT);														// predicted position

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );		

		// Write predicted positions to global memory
		predicted_pos_[index].x = p_pos_zindex.x;
		predicted_pos_[index].y = p_pos_zindex.y;
		predicted_pos_[index].z = p_pos_zindex.z;
		predicted_pos_[index].w = p_pos_zindex.w;
	}
}

__global__
	void ComputePredictedDensityAndPressureBlocksKernelPCISPH(float* density_error_, float* predicted_density_, float* corr_pressure_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_predicted_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float predicted_density = initialMass * kernelSelf;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_predicted_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_predicted_pos_zindex = tex1Dfetch( texture_predicted_pos, index );
		//smoothedColor = predicted_density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_predicted_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_predicted_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_predicted_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_predicted_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_predicted_pos_zindex[tid] = tex1Dfetch( texture_predicted_pos, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new densities from new copied neighbors in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_predicted_pos_zindex, neigh_predicted_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_predicted_pos_zindex[j].w != p_predicted_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
					distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

					//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					predicted_density += (distance * initialMass );
					//smoothedColor += initialMass * distance * neighInvDensities[j];
					//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		const float density_error = fmaxf(0.0f, predicted_density - fluidRestDensity );
		density_error_[index] = density_error;
		corr_pressure_[index] += fmaxf( 0.0f, density_error * densityErrorFactor );
		predicted_density_[index] = predicted_density;
		
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void ComputePredictedDensityAndPressureBlocksKernelPCISPHWallWeight(float* density_error_, float* predicted_density_, float* corr_pressure_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_predicted_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint	  particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint   currStartIndex;
	__shared__ uint   currNumParticles;
	__shared__ uint   currBlockIndex;
	__shared__ uint   someMaxValue;

	float predicted_density = initialMass * kernelSelf;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_predicted_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_predicted_pos_zindex = tex1Dfetch( texture_predicted_pos, index );
		//smoothedColor = predicted_density / tex1Dfetch( texture_density, index );

		// Accumulate new density contribution from static wall boundaries using Wall Weight Function Method
		// TODO: use LUT as we did in cpu version??? 
		// Tips: It may be best not to use any tables on the GPU at all (see also CUDA math library), as FLOPS are increasing faster than memory bandwidth across GPU generations.
		const float distToWall = DistanceToWallDevice(p_predicted_pos_zindex);
		const float wallWeightValue = WallWeightDevice(distToWall);
		predicted_density += wallWeightValue;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_predicted_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_predicted_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_predicted_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_predicted_pos_zindex[tid].w = intMax;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_predicted_pos_zindex[tid] = tex1Dfetch( texture_predicted_pos, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new densities from new copied neighbors in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_predicted_pos_zindex, neigh_predicted_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_predicted_pos_zindex[j].w != p_predicted_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius ;
					distance = ( tmpvari >= lutSize ) ? 0.0 : lutKernelM4Table[tmpvari];

					//float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					//float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					predicted_density += (distance * initialMass );
					//smoothedColor += initialMass * distance * neighInvDensities[j];
					//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		const float density_error = fmaxf(0.0f, predicted_density - fluidRestDensity );
		density_error_[index] = density_error;
		corr_pressure_[index] += fmaxf( 0.0f, density_error * densityErrorFactor );
		predicted_density_[index] = predicted_density;
		
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void ComputeCorrectivePressureForceBlocksKernelPCISPH(float4* corr_pressure_force_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4  neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint    particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint    currStartIndex;
	__shared__ uint    currNumParticles;
	__shared__ uint    currBlockIndex;
	__shared__ uint    someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure = 0.0f;
	float4 pcorr_pressure_force = make_float4( 0.0, 0.0, 0.0, 0.0 );

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_corr_pressure = tex1Dfetch( texture_corr_pressure, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_corr_pressure[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_corr_pressure[tid] = tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					float tmpvarf = lutKernelPressureGradientTable[dist_lut];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					// sum up pressure force according to Monaghan
					tmpvarf = (p_corr_pressure + neigh_corr_pressure[j] )* invfluidRestDensity * invfluidRestDensity;
					pcorr_pressure_force -= kernelGradient * tmpvarf * initialMass2;
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// Write updated corrective pressure forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		corr_pressure_force_[index] = pcorr_pressure_force;
	}

}

__global__
	void ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPH(float4* corr_pressure_force_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4  neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint    particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint    currStartIndex;
	__shared__ uint    currNumParticles;
	__shared__ uint    currBlockIndex;
	__shared__ uint    someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure = 0.0f;
	float4 pcorr_pressure_force = make_float4( 0.0, 0.0, 0.0, 0.0 );

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_corr_pressure = tex1Dfetch( texture_corr_pressure, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_corr_pressure[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_corr_pressure[tid] = tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					float tmpvarf = lutKernelPressureGradientTable[dist_lut];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					// sum up pressure force according to Monaghan
					tmpvarf = (p_corr_pressure + neigh_corr_pressure[j] )* invfluidRestDensity * invfluidRestDensity;
					pcorr_pressure_force -= kernelGradient * tmpvarf * initialMass2;
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// Write updated corrective pressure forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		corr_pressure_force_[index] = pcorr_pressure_force;
	}

}

__global__
	void ComputeCorrectivePressureForceBlocksKernelPCISPHTwoWayCoupling(float4* corr_pressure_force_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4  neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	   neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint    particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint    currStartIndex;
	__shared__ uint    currNumParticles;
	__shared__ uint    currBlockIndex;
	__shared__ uint    someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex			= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure		= 0.0f;
	float4 pcorr_pressure_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
	int	   p_type				= 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex	= tex1Dfetch( texture_pos_zindex, index );
		p_corr_pressure = tex1Dfetch( texture_corr_pressure, index);
		p_type			= tex1Dfetch(texture_type, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_corr_pressure[tid] = 0.0;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_corr_pressure[tid]	= tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch( texture_type, neighStartIndex + i );
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2 && neigh_type[j] == 2)
					{
						// here we only calculate pressure force from inner liquid
						float tmpvarf = lutKernelPressureGradientTable[dist_lut];
						float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

						// sum up pressure force according to Monaghan
						tmpvarf = (p_corr_pressure + neigh_corr_pressure[j] )* invfluidRestDensity * invfluidRestDensity;
						pcorr_pressure_force -= kernelGradient * tmpvarf * initialMass2;
					} 
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// Write updated corrective pressure forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		corr_pressure_force_[index] = pcorr_pressure_force;
	}

}

__global__
	void ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPHTwoWayCoupling(float4* corr_pressure_force_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4  neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_density[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	   neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint    particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint    currStartIndex;
	__shared__ uint    currNumParticles;
	__shared__ uint    currBlockIndex;
	__shared__ uint    someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex						= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure					= 0.0f;
	float  p_previous_predicted_density		= 0.0f;
	float  p_weighted_vol					= 0.0f;
	float4 pcorr_pressure_boundary_force	= make_float4( 0.0, 0.0, 0.0, 0.0 );
	int	   p_type							= 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex					= tex1Dfetch( texture_pos_zindex, index );
		p_corr_pressure					= tex1Dfetch( texture_corr_pressure, index);
		neigh_weighted_volume[tid]		= tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
		p_previous_predicted_density	= tex1Dfetch( texture_previous_predicted_density, index);
		p_weighted_vol					= tex1Dfetch( texture_weighted_volume, index);
		p_type							= tex1Dfetch(texture_type, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x  = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y  = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z  = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w  = intMax;
		neigh_corr_pressure[tid] = 0.0;
		neigh_density[tid]		 = 0.0;
		neigh_type[tid]			 = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_corr_pressure[tid]	= tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
			neigh_density[tid]		    = tex1Dfetch( texture_predicted_density, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch( texture_type, neighStartIndex + i );
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2)
					{
						if (neigh_type[j] == 2)
						{
							// For each liquid particle, we add pressure force from inner liquid
							float tmpvarf = lutKernelPressureGradientTable[dist_lut];
							float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

							// sum up pressure force according to Monaghan
							tmpvarf = (p_corr_pressure + neigh_corr_pressure[j] )* invfluidRestDensity * invfluidRestDensity;
							pcorr_pressure_boundary_force -= kernelGradient * tmpvarf * initialMass2;
						}
						else if (neigh_type[j] == 0 || neigh_type[j] == 6)	
						{
							// For each liquid particle, we add boundary forces from rigid particles & frozen boundary particles
							// versatile method E.q(9)
							pcorr_pressure_boundary_force	+= CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_previous_predicted_density, p_corr_pressure, p_weighted_vol);
						}
					} 
					else if (p_type == 0)
					{
						if (neigh_type[j] == 0)
						{
							/*
							// For each rigid particle, we add forces from rigid particles of other rigid bodies
							// "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							const float overlap = 2.0f * d_const_particle_radius - dist;
							// only calculate forces between rigid particles if they belong to different rigid body
							if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
							{
								// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
								// here r_ij = p_j - p_i  v_ij = v_j - v_i
								pcorr_pressure_boundary_force			+= CalculateSpringForce_d(dist, overlap, (neigh_pos_zindex[j] - p_pos_zindex) ) + CalculateDampingForce_d( neigh_vel[j] - p_vel);
							}
							*/
						} 
						else if (neigh_type[j] == 2)
						{
							// For each rigid particle, we add forces from liquid particles using the latter's corrected pressure
							// versatile method E.q(9)
							pcorr_pressure_boundary_force			-= CalculateBoundaryFluidPressureForceDevice(dist_lut, neigh_pos_zindex[j], p_pos_zindex, neigh_density[j], neigh_corr_pressure[j], neigh_weighted_volume[j]);

							//TODO: add boundary fluid friction force += ;

						}	
					}
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// Write updated corrective pressure forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		corr_pressure_force_[index] = pcorr_pressure_boundary_force;
	}

}

__global__
	void ComputeCorrectiveBoundaryFluidForceBlocksKernelPCISPH(float4* dynamic_boundary_force_, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4  neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4  neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float   neigh_density[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	   neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint	   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint	   currStartIndex;
	__shared__ uint    currNumParticles;
	__shared__ uint    currBlockIndex;
	__shared__ uint    someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex						= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel							= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_boundary_fluid_pressure_force  = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_boundary_fluid_friction_force  = make_float4( 0.0, 0.0, 0.0, 0.0 ); 
	float  p_corr_pressure					= 0.0f;
	float  p_density						= 0.0f;
	float  p_weighted_vol					= 0.0f;
	int	   p_type							= 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex					= tex1Dfetch( texture_pos_zindex, index );
		p_vel							= tex1Dfetch( texture_vel, index );
		p_corr_pressure					= tex1Dfetch( texture_corr_pressure, index);
		p_density						= tex1Dfetch( texture_predicted_density, index);
		p_weighted_vol					= tex1Dfetch( texture_weighted_volume, index);
		p_boundary_fluid_pressure_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
		p_boundary_fluid_friction_force = make_float4( 0.0, 0.0, 0.0, 0.0 );
		p_type							= tex1Dfetch(texture_type, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_corr_pressure[tid] = 0.0;
		neigh_density[tid]		 = 0.0;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_corr_pressure[tid]	= tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
			neigh_weighted_volume[tid]  = tex1Dfetch(texture_weighted_volume, neighStartIndex + i );
			neigh_density[tid]		    = tex1Dfetch( texture_predicted_density, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2 && (neigh_type[j] == 0 || neigh_type[j] == 6) )
					{
						// For each liquid particle, we add forces from rigid particles & frozen boundary particles			
						// versatile method E.q(9)
						p_boundary_fluid_pressure_force	  += CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_density, p_corr_pressure, p_weighted_vol);
						//p_boundary_fluid_friction_force += ;
					} 
					else if (p_type == 0 && neigh_type[j] == 2)
					{
						// For each rigid particle, we add forces from liquid particles using the latter's corrected pressure
						// versatile method E.q(10)
						p_boundary_fluid_pressure_force	  -= CalculateBoundaryFluidPressureForceDevice(dist_lut, neigh_pos_zindex[j], p_pos_zindex, neigh_density[j], neigh_corr_pressure[j], neigh_weighted_volume[j]);
						//p_boundary_fluid_friction_force -= ;
					}
					
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// Write updated corrective pressure forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		dynamic_boundary_force_[index]	= p_boundary_fluid_pressure_force + p_boundary_fluid_friction_force;
	}

}

__global__
	void TimeIntegrationBlocksKernelPCISPHPureFluid( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 )
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;

	if( index < PCOUNT )
	{
		float4 p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		float4 p_vel = tex1Dfetch( texture_vel, index );
		float4 p_force = tex1Dfetch(texture_static_force, index) + tex1Dfetch(texture_corr_pressure_force, index);
		
		// update particle positions
		p_vel += p_force * invinitialMass * deltaT;
		
		p_pos_zindex += p_vel * deltaT;

		// TODO: In PCISPH, how do we handle boundary force calculation? 
		// Where should put the boundary forces?

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

		//TODO: can we move z-index updating to here?
		// refer to void UnifiedPhysics::timeIntegration(const int i)
		//p_pos_zindex.w = ;
		
		array_vel_[index] = p_vel;
		array_pos_[index] = p_pos_zindex;
	}
}

__global__
	void TimeIntegrationBlocksKernelStaticBoundariesPCISPH( float4* array_pos_, float4* array_vel_, uint* indices1, uint* indices2 )
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;

	if( index < PCOUNT)
	{
		float4 p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		float4 p_vel = tex1Dfetch( texture_vel, index );
		float4 p_force = tex1Dfetch(texture_static_force, index) + tex1Dfetch(texture_corr_pressure_force, index);
		
		int type = tex1Dfetch(texture_type, index);
		if (type == 2)
		{
			// update particle positions
			p_vel += p_force * invinitialMass * deltaT;

			p_pos_zindex += p_vel * deltaT;

			// TODO: In PCISPH, how do we handle boundary force calculation? 
			// Where should put the boundary forces?

			// Handle collisions
			CollisionHandlingBox( p_pos_zindex, p_vel );

			//TODO: can we move z-index updating to here?
			// refer to void UnifiedPhysics::timeIntegration(const int i)
			//p_pos_zindex.w = ;

			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;
		}
		else
		{
			// boundary particles
			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;
		}

	}
}

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep1( float4* array_pos_, float4* array_vel_, uint* grid_ancillary_array, uint* indices2 )
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;
	int type = tex1Dfetch(texture_type, index);

	if( index < PCOUNT )
	{
		// we only integrate fluid particles
		float4 p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		float4 p_vel = tex1Dfetch( texture_vel, index );

		if (type == 2)
		{
			float4 p_force = tex1Dfetch(texture_static_force, index) + tex1Dfetch(texture_corr_pressure_force, index);

			// update particle positions
			p_vel += p_force * invinitialMass * deltaT;
			p_pos_zindex += p_vel * deltaT;
			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;	
		} 
		else
		{
			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;	
		}
	}
}

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep2( float4* array_pos_, float4* array_vel_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 n_i_c = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float sum_w_ib_c = 0.0;
	float sum_middle_term = 0.0;
	float4 normalized_normal = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 corrected_pos = make_float4( 0.0, 0.0, 0.0, 0.0 );
	bool isPenetrate = false;
	int	   p_type	    = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_type							= tex1Dfetch(texture_type, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	
	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = array_pos_[neighStartIndex + i];
			neigh_type[tid]		  = tex1Dfetch( texture_type, neighStartIndex + i );
		}

		__syncthreads();

		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (dist < spacing && (p_type == 2 && neigh_type[j] == 6) )	// for liquid particles with at least one boundary particle within its support radius
					{
						isPenetrate = true;
						float w_ib_c = fmaxf(0.0f, (spacing-dist)/spacing);
						const float4 n_b = make_float4(0.0, 1.0, 0.0, 0.0);	// In this case, we fixed n_b's value as pointing up (In the comparison demo, we only use one horizontal plane)
						n_i_c += n_b * w_ib_c;
						sum_middle_term += w_ib_c * (spacing-dist);
					} 
				}
			}

			//ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	// compute world collision (11) and (12) from paper "Boundary handling and adaptive time-stepping for PCISPH"
	// Note: For the paper demo comparison, we set n_b = (0, 1, 0) for simplicity 
	// we need another neighbor search here
	if( tid < currNumParticles && index < PCOUNT && isPenetrate)
	{
		float4 p_vel = array_vel_[index];
		p_pos_zindex = array_pos_[index];

		if (p_type == 2)
		{
			normalized_normal = getNormalizedVec(n_i_c);
			p_pos_zindex += normalized_normal * sum_w_ib_c * sum_middle_term;
			float4 vel_tangential = p_vel - normalized_normal * getDotProduct(p_vel, normalized_normal); 
			const float epsilon = 0.5f;
			p_vel = vel_tangential * epsilon;

			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;
		} 
		else
		{
			array_vel_[index] = p_vel;
			array_pos_[index] = p_pos_zindex;
		}
	}
}

__global__
	void TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep3( float4* array_pos_, float4* array_vel_, uint* grid_ancillary_array, uint* indices2 )
{
	uint index = blockDim.x * blockIdx.x + threadIdx.x;
	int type = tex1Dfetch(texture_type, index);

	if( type == 2 && index < PCOUNT )
	{
		// Handle collisions
		CollisionHandlingBox( array_pos_[index], array_vel_[index] );
	}
}

__global__
	void CalculateCorrectedDensityPressureInBlocksKernelSPHWallWeight(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint   currStartIndex;
	__shared__ uint	  currNumParticles;
	__shared__ uint   currBlockIndex;
	__shared__ uint   someMaxValue;

	float density = initialMass * kernelSelf;
	int	  p_type  = -1;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_type = tex1Dfetch(texture_type, index);
		//smoothedColor = density / tex1Dfetch( texture_density, index );

		// Accumulate new density contribution from static wall boundaries using Wall Weight Function Method
		// TODO: use LUT as we did in cpu version??? 
		// Tips: It may be best not to use any tables on the GPU at all (see also CUDA math library), as FLOPS are increasing faster than memory bandwidth across GPU generations.
		const float distToWall = DistanceToWallDevice(p_pos_zindex);
		const float wallWeightValue = WallWeightDevice(distToWall);
		density += wallWeightValue;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_weighted_volume[tid]	= tex1Dfetch(texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neighInvDensities[tid]	= 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new liquid densities from new copied neighbors in shared memory
		if( tid < currNumParticles && p_type == 2)
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius ;
					float kernelVaule = ( dist_lut >= lutSize ) ? 0.0 : lutKernelM4Table[dist_lut];

					if (neigh_type[j] == 2)
					{
						// sum up contribution from liquid neighbors
						density += (kernelVaule * initialMass );
						//smoothedColor += initialMass * distance * neighInvDensities[j];
						//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
					} 
					else if (neigh_type[j] == 0 || neigh_type[j] == 6)
					{
						// sum up contribution from rigid neighbors or frozen neighbors
						density += (kernelVaule * fluidRestDensity * neigh_weighted_volume[j]);
					}
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	// In contrast to cpu version, here we still store rigid particle's density & pressure 
	// (TODO: idea-> view rigid particle as liquid particle and then correct its position according to rigidbody dynamics)
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		array_pressure_[index] = fmaxf( 0.0f, (density * invfluidRestDensity - 1.0f) * fluidRestDensity * fluidGasConstant );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void CalculateCorrectedDensityPressureInBlocksKernelWCSPHWallWeight(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float density = initialMass * kernelSelf;
	int	  p_type = -1;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_type = tex1Dfetch(texture_type, index);
		//smoothedColor = density / tex1Dfetch( texture_density, index );

		// Accumulate new density contribution from static wall boundaries using Wall Weight Function Method
		// TODO: use LUT as we did in cpu version??? 
		// Tips: It may be best not to use any tables on the GPU at all (see also CUDA math library), as FLOPS are increasing faster than memory bandwidth across GPU generations.
		const float distToWall = DistanceToWallDevice(p_pos_zindex);
		const float wallWeightValue = WallWeightDevice(distToWall);
		density += wallWeightValue;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_weighted_volume[tid]	= tex1Dfetch(texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neighInvDensities[tid]	= 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new liquid densities from new copied neighbors in shared memory
		if( tid < currNumParticles && p_type == 2)
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius ;
					float kernelVaule = ( dist_lut >= lutSize ) ? 0.0 : lutKernelM4Table[dist_lut];

					if (neigh_type[j] == 2)
					{
						// sum up contribution from liquid neighbors
						density += (kernelVaule * initialMass );
						//smoothedColor += initialMass * distance * neighInvDensities[j];
						//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
					} 
					else if (neigh_type[j] == 0 || neigh_type[j] == 6)
					{
						// sum up contribution from rigid neighbors or frozen neighbors
						density += (kernelVaule * fluidRestDensity * neigh_weighted_volume[j]);
					}
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	const float b_i = fluidRestDensity * fluidGasConstantWCSPH / gamma;
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		const float powGamma = powf(density * invfluidRestDensity, gamma);
		array_pressure_[index] = fmaxf( 0.0f, (powGamma - 1.0f) * b_i );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}


// Versatile coupling method
__global__
	void CalculateWeightedVolumeInBlocksKernel(float* array_weighted_volume_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint   currStartIndex;
	__shared__ uint	  currNumParticles;
	__shared__ uint   currBlockIndex;
	__shared__ uint   someMaxValue;

	float inverse_volume = 0.0f;
	int	  p_type = -1;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;

	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_type		 = tex1Dfetch(texture_type, index);
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_type[tid] = tex1Dfetch(texture_type, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
		if( tid < currNumParticles && (p_type == 0 || p_type == 6) )		
		{
			// Accumulate new densities from new copied neighbors in shared memory
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w && (neigh_type[j] == 0 || neigh_type[j] == 6) )
				{
					uint dist_lut = sqrtf(distanceSquare) * lutSize * invglobalSupportRadius ;
					float kernelValue = ( dist_lut >= lutSize ) ? 0.0 : lutKernelM4Table[dist_lut];
					// for real boundary neighbor particles
					inverse_volume += kernelValue;				
				}
			}
		}

		__syncthreads();
	}

	if( tid < currNumParticles && index < PCOUNT )
	{
		if (p_type == 0 || p_type == 6)
		{
			// for rigid boundary particles
			// Implementation note: In order to enforce a volume ratio of 1, should be multiplied by 0.7f
			// page 65 E.q(5.9) from thesis "Particle-based Simulation of Large Bodies of Water with Bubbles, Spray and Foam"
			if (inverse_volume >= 0.1f / fluidRestVolume)
			{
				array_weighted_volume_[index] = 0.7f / inverse_volume;		// NOTE: a trick to make the value of array_weighted_volume_[index] below 7 * fluidRestVolume
			}
			else
			{
				array_weighted_volume_[index] = fluidRestVolume;			
			}			
		} 
		else
		{
			array_weighted_volume_[index] = fluidRestVolume; 
		}
	}
}

__global__
	void CalculateCorrectedDensitiesInBlocksKernelSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint   currStartIndex;
	__shared__ uint   currNumParticles;
	__shared__ uint   currBlockIndex;
	__shared__ uint   someMaxValue;

	float density = initialMass * kernelSelf;
	int	  p_type = -1;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	//int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_type = tex1Dfetch(texture_type, index);
		//smoothedColor = density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_weighted_volume[tid]	= tex1Dfetch(texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neighInvDensities[tid]	= 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new liquid densities from new copied neighbors in shared memory
		if( tid < currNumParticles && p_type == 2)
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius ;
					float kernelVaule = ( dist_lut >= lutSize ) ? 0.0 : lutKernelM4Table[dist_lut];

					if (neigh_type[j] == 2)
					{
						// sum up contribution from liquid neighbors
						density += (kernelVaule * initialMass );
						//smoothedColor += initialMass * distance * neighInvDensities[j];
						//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
					} 
					else if (neigh_type[j] == 0 || neigh_type[j] == 6)
					{
						// sum up contribution from rigid neighbors or frozen neighbors
						density += (kernelVaule * fluidRestDensity * neigh_weighted_volume[j]);
					}
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		array_pressure_[index] = fmaxf( 0.0f, (density * invfluidRestDensity - 1.0f) * fluidRestDensity * fluidGasConstant );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void CalculateCorrectedDensitiesInBlocksKernelWCSPH(float* array_pressure_, float* array_density, float4* array_smoothcolor_normal, 
	uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];	
	__shared__ float  neighInvDensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	float density = initialMass * kernelSelf;
	int	  p_type = -1;
	//float smoothedColor;
	//float4 normal = make_float4(0.0, 0.0, 0.0, 0.0);

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int offset = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( N < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
	}
	particleCountBlocks[tid] = neighNumParticles;

	__syncthreads();

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_type = tex1Dfetch(texture_type, index);
		//smoothedColor = density / tex1Dfetch( texture_density, index );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_weighted_volume[tid] = fluidRestVolume;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_weighted_volume[tid] = tex1Dfetch(texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid] = tex1Dfetch(texture_type, neighStartIndex + i );
			//neighInvDensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
		}

		__syncthreads();

		// Accumulate new liquid densities from new copied neighbors in shared memory
		if( tid < currNumParticles && p_type == 2)
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius ;
					float kernelVaule = ( dist_lut >= lutSize ) ? 0.0 : lutKernelM4Table[dist_lut];

					if (neigh_type[j] == 2)
					{
						// sum up contribution from liquid neighbors
						density += (kernelVaule * initialMass );
						//smoothedColor += initialMass * distance * neighInvDensities[j];
						//normal +=  kernelGradient * (initialMass * neighInvDensities[j]);
					} 
					else if (neigh_type[j] == 0 || neigh_type[j] == 6)
					{
						// sum up contribution from rigid neighbors or frozen neighbors
						density += (kernelVaule * fluidRestDensity * neigh_weighted_volume[j]);
					}
				}
			}
		}

		__syncthreads();
	}

	//__syncthreads();

	// Write updated densities to global memory
	const float b_i = fluidRestDensity * fluidGasConstantWCSPH / gamma;
	if( tid < currNumParticles && index < PCOUNT )
	{
		array_density[index] = density;
		const float powGamma = powf(density * invfluidRestDensity, gamma);
		array_pressure_[index] = fmaxf( 0.0f, (powGamma - 1.0f) * b_i );
		//array_smoothcolor_normal[index].w = smoothedColor;
		//array_smoothcolor_normal[index] = normal;	
	}
}

__global__
	void CalculateForcesPerParticleInBlocksKernelSPHVersatileCoupling( float4* array_pos_, float4* array_vel_,
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 pforce = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure = 0.0f;
	int	   p_type = -1;
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		p_pressure = tex1Dfetch( texture_pressure, index);
		invdensity = 1.0  / tex1Dfetch( texture_density, index );
		p_type = tex1Dfetch(texture_type, index);
		pforce = make_float4( 0.0, 0.0, 0.0, 0.0 );
		pVol = initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			neigh_type[tid] = tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					if (p_type == 2)
					{
						// For each liquid particle, we add forces from (1) liquid particles (2) rigid particles			

						if (neigh_type[j] == 2)
						{
							// (1) pressure force & viscosity force & cohesion force
							pforce += CalculatePressureForce(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_pressure, invdensity, neigh_pressure[j], neigh_invdensities[j]);
							pforce += CalculateViscosityForce(dist_lut, p_vel, neigh_vel[j], pVol, neigh_invdensities[j]);
							// TODO: add cohesion force

						}
						else if (neigh_type[j] == 0)
						{
							// (2) versatile method
						}

					} 
					else if (p_type == 0)
					{
						// For each rigid particle, we add forces from (1) liquid particles (2) rigid particles

						if (neigh_type[j] == 2)
						{
							// (1) versatile method
						}
						else if (neigh_type[j] == 0)
						{
							// (2) "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							// only calculate forces between rigid particles if they belong to different rigid body
							//const float overlap = 2.0f * d_const_particle_radius - dist;
						} 		

					}

					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					// TODO: calculate temperature diffusion change per time for melting phenomenon


					// TODO: we may add other controlling forces in here

				}
			}
		}

		//__syncthreads();	// TODO: it seems that we do not need to synchronize in here
	}

	//__syncthreads();		// TODO: it seems that we do not need to synchronize in here

	if( tid < currNumParticles)
	{		
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			pforce.y -= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			pforce += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);
		} 
		else
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles

		}

		// Update particle positions & velocities
		p_vel += (pforce * invinitialMass * deltaT );
		p_pos_zindex += (p_vel * deltaT);

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
		CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

		//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
	}
	//__syncthreads();	// TODO: it seems that we do not need to synchronize in here

	// Write updated liquid positions to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		array_pos_[index] = p_pos_zindex;
		array_vel_[index] = p_vel;
		//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
	}
}

__global__
	void CalculateForcesPerParticleInBlocksKernelWCSPHVersatileCoupling( float4* array_pos_, float4* array_vel_, 
	float4* array_smoothcolor_normal, uint* indices1, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bdim = blockDim.x;
	const unsigned int bid = blockIdx.x;

	unsigned int index = bid * bdim + tid;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel = make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure = 0.0f;
	float4 pforce_density = make_float4( 0.0, 0.0, 0.0, 0.0 );
	//float3 normal = make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex = tex1Dfetch( texture_pos_zindex, index );
		p_vel = tex1Dfetch( texture_vel, index );
		p_pressure = tex1Dfetch( texture_pressure, index);
		pforce_density = make_float4( 0.0, 0.0, 0.0, tex1Dfetch( texture_density, index ) );
		invdensity = 1.0  / pforce_density.w;
		pVol = initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = max( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid] = tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid] = tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid] = tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid] = 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			//neigh_smoothedColor[tid] = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distance = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );

				if( distance < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					tmpvari = sqrtf(distance) * lutSize * invglobalSupportRadius;
					if( tmpvari > lutSize )
						tmpvari = lutSize;
					float tmpvarf = lutKernelPressureGradientTable[tmpvari];
					float4 kernelGradient = ( p_pos_zindex - neigh_pos_zindex[j] ) * tmpvarf;

					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					tmpvarf = lutKernelViscosityLapTable[tmpvari];

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

					pforce_density -= ((p_vel - neigh_vel[j]) * tmpVis *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#else
					pforce_density -= ((p_vel - neigh_vel[j]) * fluidViscosityConstant *
						pVol * tmpvarf * initialMass * neigh_invdensities[j]);
#endif


					tmpvarf = p_pressure * invdensity * invdensity + 
						neigh_pressure[j] * neigh_invdensities[j] * neigh_invdensities[j];
					pforce_density -= kernelGradient * tmpvarf * initialMass2;	
				}
			}

			ii += bdim;
		}

		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		// Handle boundary forces
		pforce_density.y -= initialMassIntoGravityConstant;
		pforce_density += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

		// Update particle positions & velocities
		p_vel += (pforce_density * invinitialMass * deltaTWCSPH );
		p_pos_zindex += (p_vel * deltaTWCSPH);

		// Handle collisions
		CollisionHandlingBox( p_pos_zindex, p_vel );

#ifdef SPH_DEMO_SCENE_2
		CollisionHandlingCylinder( p_pos_zindex, p_vel );   
#endif

		//p_pos_zindex.w = CalcIndex( scales.x * p_pos_zindex.x, scales.y * p_pos_zindex.y, scales.z * p_pos_zindex.z );
	}
	__syncthreads();

	// Write updated positions to global memory
	if( tex1Dfetch(texture_type, index) == 2 && tid < currNumParticles && index < PCOUNT )
	{		
		array_pos_[index] = p_pos_zindex;
		array_vel_[index] = p_vel;
		//array_smoothcolor_normal[index] = make_float4( normal.x, normal.y, normal.z, 0.0 );
	}
}

__global__
	void CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH( float4* static_force_, 
																		uint* grid_ancillary_array, 
																		uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_density[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint	  particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex						= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel							= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure					= 0.0f;
	float  p_previous_predicted_density		= 0.0f;
	float  p_weighted_vol					= 0.0f;
	int	   p_type							= 0;
	//float3 normal							= make_float3( 0.0, 0.0, 0.0 );	

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex					= tex1Dfetch( texture_pos_zindex, index );
		p_vel							= tex1Dfetch( texture_vel, index );
		p_corr_pressure					= tex1Dfetch( texture_previous_corr_pressure, index);
		p_previous_predicted_density	= tex1Dfetch( texture_previous_predicted_density, index);
		p_weighted_vol					= tex1Dfetch( texture_weighted_volume, index);
		p_type							= tex1Dfetch( texture_type, index);
		p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	// Add viscosity force
	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x		= 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y		= 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z		= 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w		= intMax;
		neigh_vel[tid].x			= 0.0;
		neigh_vel[tid].y			= 0.0;
		neigh_vel[tid].z			= 0.0;
		neigh_vel[tid].w			= 0.0;
		neigh_corr_pressure[tid]	= 0.0;
		neigh_density[tid]			= 0.0;
		neigh_type[tid]				= -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_corr_pressure[tid]	= tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
			neigh_weighted_volume[tid]  = tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
			neigh_density[tid]		    = tex1Dfetch( texture_predicted_density, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid]  = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute static & dynamic external forces (excluding pressure force) using texture attributes and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2)
					{
						if (neigh_type[j] == 2)
						{
							// For each liquid particle, we add viscosity force from other liquid particles	
							// viscosity force & cohesion force
							p_static_force	+= CalculateViscosityForcePCISPH(dist_lut, p_vel, neigh_vel[j], fluidRestVolume);			
							
							// surface tension force : cohesion force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceTensionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j]);
							}
						} 
						else if (neigh_type[j] == 0 || neigh_type[j] == 6)	
						{
							// For each liquid particle, we add forces from rigid particles & frozen boundary particles			
							// versatile method E.q(9)
							p_static_force	+= CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_previous_predicted_density, p_corr_pressure, p_weighted_vol);
							//TODO: add boundary fluid friction force += ;

							// surface tension force : adhesion force according to E.q(6) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceCohesionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j], neigh_weighted_volume[j]);
							}		
						}
					} 
					else if (p_type == 0)
					{
						if (neigh_type[j] == 0)	// TODO: add neigh_type[j] == 6
						{
							// For each rigid particle, we add forces from rigid particles of other rigid bodies
							// "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							const float overlap = 2.0f * d_const_particle_radius - dist;
							// only calculate forces between rigid particles if they belong to different rigid body
							if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
							{
								// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
								// here r_ij = p_j - p_i  v_ij = v_j - v_i
								p_static_force				 += CalculateSpringForce_d(dist, overlap, (neigh_pos_zindex[j] - p_pos_zindex) ) + CalculateDampingForce_d( neigh_vel[j] - p_vel);
							}
						} 
						else if (neigh_type[j] == 2)
						{
							// For each rigid particle, we add forces from liquid particles using the latter's corrected pressure
							// versatile method E.q(10)
							p_static_force -= CalculateBoundaryFluidPressureForceDevice(dist_lut, neigh_pos_zindex[j], p_pos_zindex, neigh_density[j], neigh_corr_pressure[j], neigh_weighted_volume[j]);
							
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force += CalculateSurfaceCohesionForcePCISPHDevice(dist_lut, dist, neigh_pos_zindex[j], p_pos_zindex, p_weighted_vol);
							}						

							//TODO: add boundary fluid friction force += ;
						}	
					}

					// The following attributes should be accumulated to static external force
					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * fluidRestVolume;

					// TODO: calculate temperature diffusion change per time for melting phenomenon

					// TODO: we may add other controlling forces in here

				}
			}
		}
		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			p_static_force.y			-= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			p_static_force				+= CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

			// Add other external forces in here
		} 
		else
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles
			p_static_force			   +=  CalculateBoundaryForcePerRigidParticleDevice(p_pos_zindex, p_vel);

			// Add other external forces in here
		}
	}
	__syncthreads();

	// Write updated forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		static_force_[index]	= p_static_force;
	}
}

__global__
	void CalculateExternalForcesStaticBoundariesInBlocksKernelPCISPH( float4* static_force_, uint* grid_ancillary_array, uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint	  particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex						= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel							= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure					= 0.0f;
	float  p_previous_predicted_density		= 0.0f;
	float  p_weighted_vol					= 0.0f;
	int	   p_type							= 0;
	//float3 normal							= make_float3( 0.0, 0.0, 0.0 );	

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex					= tex1Dfetch( texture_pos_zindex, index );
		p_vel							= tex1Dfetch( texture_vel, index );
		p_corr_pressure					= tex1Dfetch( texture_previous_corr_pressure, index);
		p_previous_predicted_density	= tex1Dfetch( texture_previous_predicted_density, index);
		p_weighted_vol					= tex1Dfetch( texture_weighted_volume, index);
		p_type							= tex1Dfetch( texture_type, index);
		p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	// Add viscosity force
	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x		= 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y		= 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z		= 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w		= intMax;
		neigh_vel[tid].x			= 0.0;
		neigh_vel[tid].y			= 0.0;
		neigh_vel[tid].z			= 0.0;
		neigh_vel[tid].w			= 0.0;
		neigh_type[tid]				= -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_weighted_volume[tid]  = tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid]  = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute static & dynamic external forces (excluding pressure force) using texture attributes and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2)
					{
						if (neigh_type[j] == 2)
						{
							// For each liquid particle, we add viscosity force from other liquid particles	
							// viscosity force & cohesion force
							p_static_force	+= CalculateViscosityForcePCISPH(dist_lut, p_vel, neigh_vel[j], fluidRestVolume);			
							
							// surface tension force : cohesion force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceTensionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j]);
							}
						} 
						else if (neigh_type[j] == 6)	
						{
							// For each liquid particle, we add forces from rigid particles & frozen boundary particles			
							// versatile method E.q(9)
							p_static_force	+= CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_previous_predicted_density, p_corr_pressure, p_weighted_vol);
							//TODO: add boundary fluid friction force += ;

							// surface tension force : adhesion force according to E.q(6) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceCohesionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j], neigh_weighted_volume[j]);
							}		
						}
					} 

					// The following attributes should be accumulated to static external force
					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * fluidRestVolume;

					// TODO: calculate temperature diffusion change per time for melting phenomenon

					// TODO: we may add other controlling forces in here

				}
			}
		}
		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			p_static_force.y			-= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			p_static_force				+= CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

			// Add other external forces in here
		} 
	}
	__syncthreads();

	// Write updated forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		static_force_[index]	= p_static_force;
	}
}

__global__
	void CalculateExternalForcesWithoutBoundaryForceRigidFluidCouplingInBlocksKernelPCISPH( float4* static_force_, 
																		uint* grid_ancillary_array, 
																		uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_corr_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_volume[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_density[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint	  particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex						= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel							= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_corr_pressure					= 0.0f;
	float  p_previous_predicted_density		= 0.0f;
	float  p_weighted_vol					= 0.0f;
	int	   p_type							= 0;
	//float3 normal							= make_float3( 0.0, 0.0, 0.0 );	

	if( tid == 0 )
	{
		currBlockIndex = indices2[3*bid];
		currStartIndex = indices2[3*bid+1];
		currNumParticles = indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = grid_ancillary_array[2*N];
			neighNumParticles = grid_ancillary_array[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex					= tex1Dfetch( texture_pos_zindex, index );
		p_vel							= tex1Dfetch( texture_vel, index );
		p_corr_pressure					= tex1Dfetch( texture_previous_corr_pressure, index);
		p_previous_predicted_density	= tex1Dfetch( texture_previous_predicted_density, index);
		p_weighted_vol					= tex1Dfetch( texture_weighted_volume, index);
		p_type							= tex1Dfetch( texture_type, index);
		p_static_force					= make_float4( 0.0, 0.0, 0.0, 0.0 );
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	// Add viscosity force
	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x		= 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y		= 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z		= 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w		= intMax;
		neigh_vel[tid].x			= 0.0;
		neigh_vel[tid].y			= 0.0;
		neigh_vel[tid].z			= 0.0;
		neigh_vel[tid].w			= 0.0;
		neigh_corr_pressure[tid]	= 0.0;
		neigh_density[tid]			= 0.0;
		neigh_type[tid]				= -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_corr_pressure[tid]	= tex1Dfetch( texture_corr_pressure, neighStartIndex + i );
			neigh_weighted_volume[tid]  = tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
			neigh_density[tid]		    = tex1Dfetch( texture_predicted_density, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid]  = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute static & dynamic external forces (excluding pressure force) using texture attributes and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					// p_type == 0: rigid particles p_type == 2: liquid particles  p_type == 6 : frozen boundary particles	
					if (p_type == 2)
					{
						if (neigh_type[j] == 2)
						{
							// For each liquid particle, we add viscosity force from other liquid particles	
							// viscosity force & cohesion force
							p_static_force	+= CalculateViscosityForcePCISPH(dist_lut, p_vel, neigh_vel[j], fluidRestVolume);			
							
							// surface tension force : cohesion force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceTensionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j]);
							}
						} 
						else if (neigh_type[j] == 0 || neigh_type[j] == 6)	
						{
							// For each liquid particle, we add boundary fluid friction forces from rigid particles & frozen boundary particles			
							// similar to versatile method E.q(11)
							p_static_force	+= CalculateViscosityForcePCISPH(dist_lut, p_vel, neigh_vel[j], neigh_weighted_volume[j]);

							// surface tension force : adhesion force according to E.q(6) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force -= CalculateSurfaceCohesionForcePCISPHDevice(dist_lut, dist, p_pos_zindex, neigh_pos_zindex[j], neigh_weighted_volume[j]);
							}
						}
					} 
					else if (p_type == 0)
					{
						if (neigh_type[j] == 0 || neigh_type[j] == 6)
						{
							// For each rigid particle, we add forces from rigid particles of other rigid bodies or from static particles
							// "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							const float overlap = 2.0f * d_const_particle_radius - dist;
							// only calculate forces between rigid particles if they belong to different rigid body
							if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
							{
								// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
								// here r_ij = p_j - p_i  v_ij = v_j - v_i
								p_static_force				 += CalculateSpringForce_d(dist, overlap, (neigh_pos_zindex[j] - p_pos_zindex) ) + CalculateDampingForce_d( neigh_vel[j] - p_vel);
							}
						} 
						else if (neigh_type[j] == 2)
						{
							// For each rigid particle, we add forces from liquid particles using the latter's corrected pressure
							// versatile method E.q(10)
							p_static_force -= CalculateViscosityForcePCISPH(dist_lut, neigh_vel[j], p_vel, p_weighted_vol);
							
							if (dist >= globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
							{
								p_static_force += CalculateSurfaceCohesionForcePCISPHDevice(dist_lut, dist, neigh_pos_zindex[j], p_pos_zindex, p_weighted_vol);
							}
							
							//TODO: add boundary fluid friction force += ;

						}	
					}

					// The following attributes should be accumulated to static external force
					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * fluidRestVolume;

					// TODO: calculate temperature diffusion change per time for melting phenomenon

					// TODO: we may add other controlling forces in here

				}
			}
		}
		__syncthreads();
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			p_static_force.y			-= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			p_static_force				+= CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);

			// Add other external forces in here
		} 
		else
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles
			p_static_force			   +=  CalculateBoundaryForcePerRigidParticleDevice(p_pos_zindex, p_vel);

			// Add other external forces in here
		}
	}
	__syncthreads();

	// Write updated forces to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		static_force_[index]	= p_static_force;
	}
}

__global__
	void UpdateRigidParticleIndicesKernel(RigidBody_GPU* rigid_bodies_gpu)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < PCOUNT && tex1Dfetch(texture_type, index) == 0 )	// for each rigid particle
	{
		const uint	parent_rb_index					= tex1Dfetch(texture_parent_rb, index); 
		const uint	order_in_child_particles		= tex1Dfetch(texture_order_in_child_particles_array, index);
		rigid_bodies_gpu[parent_rb_index].rigid_particle_indices[order_in_child_particles] = index;
	}
}

__global__
	void CalculateForcesBetweenRigidFluidParticlesInBlocksKernelSPH( float4* p_force_array, 
																/*float4* array_smoothcolor_normal,*/ 
																uint* indices1, 
																uint* indices2 )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_vol[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex	= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel		= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 pforce		= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure	= 0.0f;
	float  p_density	= 0.0f;
	float  p_weighted_vol = 0.0f;
	int	   p_type		= 0;
	//float3 normal		= make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex		= indices2[3*bid];
		currStartIndex		= indices2[3*bid+1];
		currNumParticles	= indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex	= tex1Dfetch( texture_pos_zindex, index );
		p_vel			= tex1Dfetch( texture_vel, index );
		p_pressure		= tex1Dfetch( texture_pressure, index);
		p_density		= tex1Dfetch( texture_density, index);
		p_weighted_vol	= tex1Dfetch( texture_weighted_volume, index);
		pforce			= make_float4( 0.0, 0.0, 0.0, 0.0 );
		p_type			= tex1Dfetch(texture_type, index);
		invdensity		= 1.0  / p_density;
		pVol			= initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;
		neigh_weighted_vol[tid] = 0.0;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid]			= tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid]		= 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			neigh_weighted_vol[tid]     = tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid]  = array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					if (p_type == 2)
					{
						// For each liquid particle, we add forces from (1) liquid particles (2) rigid/frozen particles			

						if (neigh_type[j] == 2)
						{
							// (1) pressure force & viscosity force & cohesion force
							pforce += CalculatePressureForce(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_pressure, invdensity, neigh_pressure[j], neigh_invdensities[j]);
							pforce += CalculateViscosityForce(dist_lut, p_vel, neigh_vel[j], pVol, neigh_invdensities[j]);
							// TODO: add cohesion force

						}
						else if (neigh_type[j] == 0 || neigh_type[j] == 6)
						{
							// (2) versatile method
							pforce += CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_density, p_pressure, p_weighted_vol);
						}

					} 
					else if (p_type == 0)
					{
						// For each rigid particle, we add forces from (1) liquid particles (2) rigid particles

						if (neigh_type[j] == 2)
						{
							// (1) versatile method
							pforce -= CalculateBoundaryFluidPressureForceDevice(dist_lut, neigh_pos_zindex[j], p_pos_zindex, 1.0/neigh_invdensities[j], neigh_pressure[j], neigh_weighted_vol[j]);
						}
						else if (neigh_type[j] == 0)
						{
							// (2) "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							const float overlap = 2.0f * d_const_particle_radius - dist;
							// only calculate forces between rigid particles if they belong to different rigid body
							if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
							{
								// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
								pforce += CalculateSpringForce_d(dist, overlap, (neigh_pos_zindex[j] - p_pos_zindex) );		// here r_ij = p_j - p_i
								pforce += CalculateDampingForce_d( neigh_vel[j] - p_vel);										// here v_ij = v_j - v_i
							}
							
						} 		

					}

					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					// TODO: calculate temperature diffusion change per time for melting phenomenon


					// TODO: we may add other controlling forces in here

				}
			}
		}

		//__syncthreads();
	}

	//__syncthreads();

	// write updated particle force to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			pforce.y -= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			pforce += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);
		} 
		else if (p_type == 0)
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles
			pforce += CalculateBoundaryForcePerRigidParticleDevice(p_pos_zindex, p_vel);
		}

		p_force_array[index] = pforce;
	}

}

// TODO: need modifying
__global__
	void CalculateForcesBetweenRigidParticlesInBlocksKernelWCSPH( float4* p_force_array, 
																  /*float4* array_smoothcolor_normal,*/ 
																  uint* indices1, 
																  uint* indices2 
																  )
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ float4 neigh_pos_zindex[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float4 neigh_vel[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_pressure[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_invdensities[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ float  neigh_weighted_vol[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ int	  neigh_type[MAX_THREADS_PER_BLOCK_SPH];
	__shared__ uint   particleCountBlocks[MAX_THREADS_PER_BLOCK_SPH];
	//__shared__ float neigh_smoothedColor[MAX_THREADS_PER_BLOCK_SPH];

	__shared__ uint currStartIndex;
	__shared__ uint currNumParticles;
	__shared__ uint currBlockIndex;
	__shared__ uint someMaxValue;

	uint neighStartIndex = 0;
	uint neighNumParticles = 0;
	int i = 0, j = 0, k = 0;
	int ii = 0, jj = 0, kk = 0;

	float4 p_pos_zindex		= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 p_vel			= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float4 pforce			= make_float4( 0.0, 0.0, 0.0, 0.0 );
	float  p_pressure		= 0.0f;
	float  p_density		= 0.0f;
	float  p_weighted_vol	= 0.0f;
	int	   p_type			= 0;
	//float3 normal			= make_float3( 0.0, 0.0, 0.0 );	
	float invdensity;
	float pVol;

	if( tid == 0 )
	{
		currBlockIndex		= indices2[3*bid];
		currStartIndex		= indices2[3*bid+1];
		currNumParticles	= indices2[3*bid+2];
	}

	__syncthreads();

	unsigned int index = currStartIndex + tid;

	int N = currBlockIndex;

	uint tmpvari = dimX * dimY;
	while( N >= tmpvari )
	{
		N -= tmpvari;
		k++;
	}

	while( N >= dimX )
	{
		N -= dimX;
		j++;
	}

	i = N;

	/*tmpvari = dimX * dimZ;
	while( N >= tmpvari )
	{
	N -= tmpvari;
	j++;
	}

	while( N >= dimZ )
	{
	N -= dimZ;
	i++;
	}

	k = N;*/

	N = tid;
	if( tid < 27 )
	{
		while( N >= 9 )
		{
			N -= 9;
			kk++;
		}
		kk--;

		while( N >= 3 )
		{
			N -= 3;
			jj++;
		}
		jj--;

		ii = N - 1;

		i = i + ii;
		j = j + jj;
		k = k + kk;

		//N = (dimX * dimZ) * j + (dimZ * i + k);
		//N = (dimX * dimY) * k + (dimY * i + j);
		N = (dimX * dimY) * k + (dimX * j + i);

		neighStartIndex = 0;
		neighNumParticles = 0;
		if( N >= 0 && N < dimX * dimY * dimZ )
		{
			neighStartIndex = indices1[2*N];
			neighNumParticles = indices1[2*N+1];
		}
		particleCountBlocks[tid] = neighNumParticles;
	}

	__syncthreads();

	if( tid < currNumParticles )
	{
		p_pos_zindex	= tex1Dfetch( texture_pos_zindex, index );
		p_vel			= tex1Dfetch( texture_vel, index );
		p_pressure		= tex1Dfetch( texture_pressure, index);
		p_density		= tex1Dfetch( texture_density, index);
		p_weighted_vol	= tex1Dfetch( texture_weighted_volume, index);
		pforce			= make_float4( 0.0, 0.0, 0.0, 0.0 );
		p_type			= tex1Dfetch(texture_type, index);
		invdensity		= 1.0  / tex1Dfetch( texture_density, index );
		pVol			= initialMass * invdensity;
	}

	// Determine M: max particles in neighbors
	if( tid == 0 )
	{
		someMaxValue = 0;
		for( i = 0; i < 27; i++ )
		{
			someMaxValue = fmaxf( someMaxValue, particleCountBlocks[i] );
		}
	}	

	__syncthreads();

	for( i = 0; i < someMaxValue; i++ )
	{
		neigh_pos_zindex[tid].x = 4*maxBoundingBox.x;
		neigh_pos_zindex[tid].y = 4*maxBoundingBox.y;
		neigh_pos_zindex[tid].z = 4*maxBoundingBox.z;
		neigh_pos_zindex[tid].w = intMax;
		neigh_vel[tid].x = 0.0;
		neigh_vel[tid].y = 0.0;
		neigh_vel[tid].z = 0.0;
		neigh_vel[tid].w = 0.0;
		neigh_pressure[tid] = 0.0;
		neigh_invdensities[tid] = 0.0;
		neigh_weighted_vol[tid] = 0.0;
		neigh_type[tid] = -1;

		// Copy a particle from neighboring blocks 0 to 26 (one per thread) to its own shared memory
		if( i < neighNumParticles && neighStartIndex + i < PCOUNT )
		{
			neigh_pos_zindex[tid]		= tex1Dfetch( texture_pos_zindex, neighStartIndex + i );
			neigh_vel[tid]				= tex1Dfetch( texture_vel, neighStartIndex + i );
			neigh_pressure[tid]			= tex1Dfetch( texture_pressure, neighStartIndex + i );
			neigh_invdensities[tid]		= 1.0 / tex1Dfetch( texture_density, neighStartIndex + i );
			neigh_weighted_vol[tid]     = tex1Dfetch( texture_weighted_volume, neighStartIndex + i );
			neigh_type[tid]				= tex1Dfetch(texture_type, neighStartIndex + i );
			//neigh_smoothedColor[tid]	= array_smoothcolor_normal[neighStartIndex + i].w;
		}

		__syncthreads();

		// Compute new forces using texture densities and neighbors copied in shared memory
		if( tid < currNumParticles )
		{
			for( j = 0; j < 27; j++ )
			{
				float distanceSquare = distanceSq( p_pos_zindex, neigh_pos_zindex[j] );
				if( distanceSquare < globalSupportRadius2 && neigh_pos_zindex[j].w != p_pos_zindex.w )
				{
					const float dist = sqrtf(distanceSquare);
					uint dist_lut = dist * lutSize * invglobalSupportRadius;
					if( dist_lut > lutSize )
						dist_lut = lutSize;

					if (p_type == 2)
					{
						// For each liquid particle, we add forces from (1) liquid particles (2) rigid/frozen particles			

						if (neigh_type[j] == 2)
						{
							// (1) pressure force & viscosity force & cohesion force
							pforce += CalculatePressureForce(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_pressure, invdensity, neigh_pressure[j], neigh_invdensities[j]);
							pforce += CalculateViscosityForce(dist_lut, p_vel, neigh_vel[j], pVol, neigh_invdensities[j]);
							// TODO: add cohesion force

						}
						else if (neigh_type[j] == 0 || neigh_type[j] == 6)
						{
							// (2) versatile method
							pforce += CalculateBoundaryFluidPressureForceDevice(dist_lut, p_pos_zindex, neigh_pos_zindex[j], p_density, p_pressure, p_weighted_vol);
						}

					} 
					else if (p_type == 0)
					{
						// For each rigid particle, we add forces from (1) liquid particles (2) rigid particles

						if (neigh_type[j] == 2)
						{
							// (1) versatile method
							pforce -= CalculateBoundaryFluidPressureForceDevice(dist_lut, neigh_pos_zindex[j], p_pos_zindex, 1.0/neigh_invdensities[j], neigh_pressure[j], neigh_weighted_vol[j]);
						}
						else if (neigh_type[j] == 0)
						{
							// (2) "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
							const float overlap = 2.0f * d_const_particle_radius - dist;
							// only calculate forces between rigid particles if they belong to different rigid body
							if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
							{
								// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
								pforce += CalculateSpringForce_d(dist, overlap, (neigh_pos_zindex[j] - p_pos_zindex) );		// here r_ij = p_j - p_i
								pforce += CalculateDampingForce_d( neigh_vel[j] - p_vel);										// here v_ij = v_j - v_i
							}

						} 		

					}

					// TODO: calculate surface normal and lap c_s
					//normal -= kernelGradient * neigh_smoothedColor[j] * initialMass * neigh_invdensities[j];


					// TODO: calculate temperature diffusion change per time for melting phenomenon


					// TODO: we may add other controlling forces in here

				}
			}
		}

		//__syncthreads();
	}

	//__syncthreads();

	// write updated particle force to global memory
	if( tid < currNumParticles && index < PCOUNT )
	{		
		if (p_type == 2)
		{
			// add gravity force to liquid particles
			// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
			// gravitational force to rigid body force & torque calculation
			pforce.y -= initialMassIntoGravityConstant;

			// Handle boundary forces exerted on liquid particles
			pforce += CalculateBoundaryForcePerLiquidParticleDevice( p_pos_zindex);
		} 
		else
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles
			pforce += CalculateBoundaryForcePerRigidParticleDevice(p_pos_zindex, p_vel);
		}

		p_force_array[index] = pforce;
	}
}

__global__ 
	void RigidBodyIntegrationKernel(const int			currIndex,
									const float4* 		particle_force,
									RigidBody_GPU*		rigid_bodies_gpu								 
									)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < RIGIDBODYCOUNT)
	{
		// Input(constant)
		const Matrix3x3_d inv_inertia_tensor_local = rigid_bodies_gpu[index].inv_inertia_local;
		const int num_particles = rigid_bodies_gpu[index].num_particles;
		const float rb_mass = rigid_bodies_gpu[index].mass;
		const float rb_inv_mass = 1.0f / rb_mass;
		const int* rb_particle_indices = rigid_bodies_gpu[index].rigid_particle_indices;

		// Intermediate
		float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		force.y -= gravityConstant * rb_mass;											

		float4 torque = make_float4(0.0f, 0.0f, 0.0f, 0.0f);			// gravity force exerts no torque
		float4 linear_momentum = rigid_bodies_gpu[index].linear_momentum;
		float4 angular_momentum = rigid_bodies_gpu[index].angular_momentum;

		// Output
		Matrix3x3_d rb_rotation_matrix = rigid_bodies_gpu[index].rotation_matrix;
		Matrix3x3_d inv_inertia_tensor_world = rigid_bodies_gpu[index].inv_inertia_world;

		float4 pos = rigid_bodies_gpu[index].pos;
		float4 vel = rigid_bodies_gpu[index].vel;
		float4 angular_vel = rigid_bodies_gpu[index].angular_velocity;
		float4 rb_quaternion = rigid_bodies_gpu[index].quaternion;
		
		// The following equations are from "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
		CalculateForceTorque_d(index, num_particles, particle_force, pos, rb_particle_indices, force, torque);

		// Equation 1 : dP/dt = F & Equation 4 : dL/dt = r x F
		UpdateMomenta_d(index, deltaT, force, torque, rb_mass, linear_momentum, angular_momentum);

		// Equation 6 : I^-1_world = R * I^-1_local * R^T  
		UpdateInverseInertiaTensor_d(index, rb_rotation_matrix, inv_inertia_tensor_local, inv_inertia_tensor_world);

		// Equation 2 : v = P/M & Equation 3 : dx/dt = v
		PerformLinearStep_d(index, rb_inv_mass, linear_momentum, deltaT, pos, vel);

		// Equation 5 : w = I^-1 * L
		PerformAngularStep_d(index, inv_inertia_tensor_world, angular_momentum, deltaT, angular_vel, rb_quaternion);

		UpdateRotationMatrix_d(rb_quaternion, rb_rotation_matrix);

		// store the update data back to the corresponding data array
		rigid_bodies_gpu[index].rotation_matrix = rb_rotation_matrix;
		rigid_bodies_gpu[index].inv_inertia_world = inv_inertia_tensor_world;
		rigid_bodies_gpu[index].angular_velocity = angular_vel;
		rigid_bodies_gpu[index].quaternion = rb_quaternion;
		rigid_bodies_gpu[index].linear_momentum = linear_momentum;
		rigid_bodies_gpu[index].angular_momentum = angular_momentum;

		rigid_bodies_gpu[index].pos = pos;
		rigid_bodies_gpu[index].vel = vel;
	}
}

__global__ 
	void RigidBodyIntegrationTwoWayCouplingKernel(const int				currIndex,
												  RigidBody_GPU*		rigid_bodies_gpu								
												  )
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < RIGIDBODYCOUNT)
	{
		// Input(constant)
		const Matrix3x3_d inv_inertia_tensor_local = rigid_bodies_gpu[index].inv_inertia_local;
		const int num_particles = rigid_bodies_gpu[index].num_particles;
		const float rb_mass = rigid_bodies_gpu[index].mass;
		const float rb_inv_mass = 1.0f / rb_mass;
		const int* rb_particle_indices = rigid_bodies_gpu[index].rigid_particle_indices;

		// Intermediate
		float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		force.y -= gravityConstant * rb_mass;											

		float4 torque = make_float4(0.0f, 0.0f, 0.0f, 0.0f);			// gravity force exerts no torque
		float4 linear_momentum = rigid_bodies_gpu[index].linear_momentum;
		float4 angular_momentum = rigid_bodies_gpu[index].angular_momentum;

		// Output
		Matrix3x3_d rb_rotation_matrix = rigid_bodies_gpu[index].rotation_matrix;
		Matrix3x3_d inv_inertia_tensor_world = rigid_bodies_gpu[index].inv_inertia_world;

		float4 pos = rigid_bodies_gpu[index].pos;
		float4 vel = rigid_bodies_gpu[index].vel;
		float4 angular_vel = rigid_bodies_gpu[index].angular_velocity;
		float4 rb_quaternion = rigid_bodies_gpu[index].quaternion;

		// The following equations are from "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
		CalculateForceTorqueTwoWayCoupling_d(index, num_particles, pos, rb_particle_indices, force, torque);

		// Equation 1 : dP/dt = F & Equation 4 : dL/dt = r x F
		UpdateMomenta_d(index, deltaT, force, torque, rb_mass, linear_momentum, angular_momentum);

		// Equation 6 : I^-1_world = R * I^-1_local * R^T  
		UpdateInverseInertiaTensor_d(index, rb_rotation_matrix, inv_inertia_tensor_local, inv_inertia_tensor_world);

		// Equation 2 : v = P/M & Equation 3 : dx/dt = v
		PerformLinearStep_d(index, rb_inv_mass, linear_momentum, deltaT, pos, vel);

		// Equation 5 : w = I^-1 * L
		PerformAngularStep_d(index, inv_inertia_tensor_world, angular_momentum, deltaT, angular_vel, rb_quaternion);

		UpdateRotationMatrix_d(rb_quaternion, rb_rotation_matrix);

		// store the update data back to the corresponding data array
		rigid_bodies_gpu[index].rotation_matrix = rb_rotation_matrix;
		rigid_bodies_gpu[index].inv_inertia_world = inv_inertia_tensor_world;
		rigid_bodies_gpu[index].angular_velocity = angular_vel;
		rigid_bodies_gpu[index].quaternion = rb_quaternion;
		rigid_bodies_gpu[index].linear_momentum = linear_momentum;
		rigid_bodies_gpu[index].angular_momentum = angular_momentum;

		rigid_bodies_gpu[index].pos = pos;
		rigid_bodies_gpu[index].vel = vel;
	}
}

__global__
	void SynRigidParticlesKernel(const int currIndex,
								 RigidBody_GPU*		rigid_bodies_gpu,
								 float4*			p_pos_array_out,
								 float4*			p_vel_array_out)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < PCOUNT)	
	{
		if (tex1Dfetch(texture_type, index) == 0)
		{	
			// for rigid particles
			const uint			parent_rb_index		= tex1Dfetch(texture_parent_rb, index); 

			// get its parent rigid body's attributes(constant in this case)
			const float4		rb_pos				= rigid_bodies_gpu[parent_rb_index].pos;
			const float4		rb_vel				= rigid_bodies_gpu[parent_rb_index].vel;
			const float4		rb_angular_vel		= rigid_bodies_gpu[parent_rb_index].angular_velocity;
			const Matrix3x3_d&	rb_rotation_matrix	= rigid_bodies_gpu[parent_rb_index].rotation_matrix;

			const float4& p_relative_pos = tex1Dfetch(texture_relative_pos, index);		
			float4 particle_pos = tex1Dfetch(texture_pos_zindex, index);
			//const float z_index = particle_pos.w;
			float4 particle_vel = tex1Dfetch(texture_vel, index);

			ApplyRotationToParticle_d(rb_rotation_matrix, 
				p_relative_pos, 
				rb_pos, 
				rb_vel, 
				rb_angular_vel, 
				particle_pos, 
				particle_vel);

			p_pos_array_out[index].x = particle_pos.x;
			p_pos_array_out[index].y = particle_pos.y;
			p_pos_array_out[index].z = particle_pos.z;
			//p_pos_array_out[index].w = z_index;

			p_vel_array_out[index].x = particle_vel.x;
			p_vel_array_out[index].y = particle_vel.y;
			p_vel_array_out[index].z = particle_vel.z;
		} 
		else if ( tex1Dfetch(texture_type, index) == 2 ) 
		{
			// for liquid particles
			p_pos_array_out[index] = tex1Dfetch(texture_pos_zindex, index);
			p_vel_array_out[index] = tex1Dfetch(texture_vel, index);
		}
	}
}

__global__
	void CalculateCorrectedForcePerParticleKernel(float4* corrected_force_array)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < PCOUNT && tex1Dfetch(texture_type, index) == 2)
	{
		// update liquid particle's force
	}
}

__global__
	void UpdateLiquidParticleKernelSPH(const int		currIndex, 
									   const float4*	p_force_array,
									   float4*			p_pos_array,
									   float4*			p_vel_array
									   )
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < PCOUNT)
	{
		// Symplectic Euler Integration: v' = v + F/m * t p' = p + v' * t
		const float damping = 0.0f;
		float4 vel = tex1Dfetch(texture_vel, index);
		float4 pos = tex1Dfetch(texture_pos_zindex, index);

		int p_type = tex1Dfetch(texture_type, index);
		if (p_type == 2)
		{
			// liquid particles
			vel += p_force_array[index] * invinitialMass * deltaT;
			pos += vel * deltaT;

			BoundaryHandlingBoxPerNonRigidParticle_d(damping, pos, vel);

			p_vel_array[index] = vel;
			p_pos_array[index] = pos;
		}
		else
		{
			// rigid particles
			p_vel_array[index] = vel;
			p_pos_array[index] = pos;
		}
	}
}

__global__
	void UpdateLiquidParticleKernelPCISPH(const int		currIndex, 
										  float4*		p_pos_array,
										  float4*		p_vel_array)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < PCOUNT)
	{
		// Symplectic Euler Integration: v' = v + F/m * t p' = p + v' * t
		const float damping = 0.0f;
		float4 vel = tex1Dfetch(texture_vel, index);
		float4 pos = tex1Dfetch(texture_pos_zindex, index);
		float4 force = tex1Dfetch(texture_static_force, index) /*+ tex1Dfetch(texture_dynamic_boundary_force, index)*/ + tex1Dfetch(texture_corr_pressure_force, index);	// TODO: use dprt.d_force instead

		int p_type = tex1Dfetch(texture_type, index);
		if (p_type == 2)
		{
			// liquid particles
			vel += force * invinitialMass * deltaT;
			pos += vel * deltaT;

			BoundaryHandlingBoxPerNonRigidParticle_d(damping, pos, vel);

			p_vel_array[index] = vel;
			p_pos_array[index] = pos;
		}
		else
		{
			// rigid particles
			p_vel_array[index] = vel;
			p_pos_array[index] = pos;
		}
	}
}
