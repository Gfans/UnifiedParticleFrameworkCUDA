#include "cudaSPH_Include.h"
#include "global.h"
#include "RigidBody.h"
#include "particles_kernel.cuh"
#include <Windows.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

cudaChannelFormatDesc uint_to_tex = cudaCreateChannelDesc<unsigned int>();

cudaChannelFormatDesc float_to_tex = cudaCreateChannelDesc<float>();
cudaArray* array_lutKernelM4;

#ifdef SPH_PROFILING_VERBOSE

float elapsedTime;
cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;

#endif

extern "C"
{
	////////////////////////////////////////////////////////////////////////////////
	//! Perform a radix sort
	//! Sorting performed in place on passed arrays.
	//!
	//! @param pData0       input and output array - data will be sorted
	//! @param pData1       additional array to allow ping pong computation
	//! @param elements     number of elements to sort
	////////////////////////////////////////////////////////////////////////////////
	void RadixSort(KeyValuePair *pData0, KeyValuePair *pData1, uint elements, uint bits)
	{
		// Round element count to total number of threads for efficiency
		uint elements_rounded_to_3072;
		int modval = elements % 3072;
		if( modval == 0 )
			elements_rounded_to_3072 = elements;
		else
			elements_rounded_to_3072 = elements + (3072 - (modval));

		// Iterate over n bytes of y bit word, using each byte to sort the list in turn
		for (uint shift = 0; shift < bits; shift += RADIX)
		{
			// Perform one round of radix sorting

			// Generate per radix group sums radix counts across a radix group
			RadixSum<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, GRFSIZE>>>(pData0, elements, 
				elements_rounded_to_3072, shift);
			// Prefix sum in radix groups, and then between groups throughout a block
			RadixPrefixSum<<<PREFIX_NUM_BLOCKS, PREFIX_NUM_THREADS_PER_BLOCK, PREFIX_GRFSIZE>>>();
			// Sum the block offsets and then shuffle data into bins
			RadixAddOffsetsAndShuffle<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, SHUFFLE_GRFSIZE>>>(pData0, 
				pData1, elements, elements_rounded_to_3072, shift); 

			// Exchange data pointers
			KeyValuePair* pTemp = pData0;
			pData0 = pData1;
			pData1 = pTemp;
		}
	}

	uint CopyParticlesCUDA(float* pos_zindex, float* vel, float* corr_pressure, float* predicted_density, int* particleType, int* activeType, dataPointers& dptr)
	{
		CUDA_SAFE_CALL( cudaMemcpy( pos_zindex,			dptr.d_pos_zindex[dptr.currIndex],			dptr.particleCountRounded * sizeof(float4),  cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( vel,				dptr.d_vel[dptr.currIndex],					dptr.particleCountRounded * sizeof(float4),  cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( corr_pressure,		dptr.d_correctionPressure,					dptr.particleCountRounded * sizeof(float),	 cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( predicted_density,	dptr.d_predictedDensity,					dptr.particleCountRounded * sizeof(float),	 cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( particleType,		dptr.d_type[dptr.currIndex],				dptr.particleCountRounded * sizeof(int),	 cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( activeType,			dptr.d_active_type[dptr.currIndex],			dptr.particleCountRounded * sizeof(int),	 cudaMemcpyDeviceToHost ) );

		return dptr.particleCount;
	}

	void RegisterGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsNone));
	}

	void UnregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	}

	void *MapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
	{
		void *ptr;
		CUDA_SAFE_CALL(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes;
		CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
			*cuda_vbo_resource));
		return ptr;
	}

	void UnmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void MallocDeviceArraysParticles(dataPointers& dptr)
	{
		uint count = dptr.finalParticleCount;
		if( count % MAX_THREADS_PER_BLOCK_SPH )
			count += ( MAX_THREADS_PER_BLOCK_SPH - count % MAX_THREADS_PER_BLOCK_SPH );
		dptr.finalParticleCountRounded = count;

		count = dptr.particleCount;
		if( count % MAX_THREADS_PER_BLOCK_SPH )
			count += ( MAX_THREADS_PER_BLOCK_SPH - count % MAX_THREADS_PER_BLOCK_SPH );
		dptr.particleCountRounded = count;

		count = dptr.finalParticleCountRounded;

		num_blocks = ceil((float)count/MAX_THREADS_PER_BLOCK_SPH);
		gridsize_indices = 2.0 * dptr.dimX * dptr.dimY * dptr.dimZ/MAX_THREADS_PER_BLOCK_SPH;	// gridsize_indices = 2 * total_blocks / MAX_THREADS_PER_BLOCK_SPH It's for launchinig kernel "ClearIndexArrays"

		std::cout << "Setting grid indices thread dimension " << gridsize_indices << std::endl;
		uint countForGrid = (count > dptr.dimX * dptr.dimY * dptr.dimZ) ? count : (dptr.dimX * dptr.dimY * dptr.dimZ);

		CUDA_SAFE_CALL( cudaMalloc( (void**)&dptr.particlesKVPair_d[0], 2*countForGrid * sizeof(uint) ) );	// WHY allocating 2*countForGrid * sizeof(uint)
		CUDA_SAFE_CALL( cudaMalloc( (void**)&dptr.particlesKVPair_d[1], 2*countForGrid * sizeof(uint) ) );

#ifndef USE_VBO_CUDA

		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_pos_zindex[0]), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_pos_zindex[1]), count*sizeof(float4) ) );

#endif

		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_filtered_pos_zindex), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_vel[0]), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_vel[1]), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_smoothcolor_normal), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_predictedPos), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_predictedVel), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_static_force), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_dynamic_boundary_force), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_correctionPressureForce), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_force), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_pressure), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_density), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_weighted_volume), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_type[0]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_type[1]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_active_type[0]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_active_type[1]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_parent_rb[0]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_parent_rb[1]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_order_in_child_particles_array[0]), count*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_order_in_child_particles_array[1]), count*sizeof(int) ) );		
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_correctionPressure), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_previous_correctionPressure), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_densityError), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_predictedDensity), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_previous_predicted_density), count*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_rigid_particle_relative_pos[0]), count*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_rigid_particle_relative_pos[1]), count*sizeof(float4) ) );

		// temporary space for storing block results of reduction
		dptr.param_num_threads_custom_reduction = MAX_THREADS_PER_BLOCK_SPH;		// TODO: use 256 threads/block or other configurations
		dptr.param_num_blocks_custom_reduction  = (dptr.finalParticleCountRounded + MAX_THREADS_PER_BLOCK_SPH - 1) / MAX_THREADS_PER_BLOCK_SPH;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_max_predicted_density_array), dptr.param_num_blocks_custom_reduction * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_max_predicted_density_value), sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.tmpArray), count*sizeof(uint) ) );

	}

	void MallocDeviceArraysRigidBody(dataPointers& dptr, std::vector<RigidBody*>& rigidbodies_h)
	{
		const uint num_rigid_bodies = rigidbodies_h.size();
		// TODO: determine the cuda grid configuration 

		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dptr.d_rigid_bodies), num_rigid_bodies*sizeof(RigidBody_GPU) ) );

	}

	void SetDeviceConstants(dataPointers& dptr)
	{
		// Please do not change the order of these functions
		SetParticleCount(dptr.particleCount);
		SetRigidBodyCount(dptr.num_rigid_bodies);
		SetGlobalSupportRadius(dptr.globalSupportRadius);
		SetDistToCenterMassCutoffApproximate(dptr.distToCenterMassCutoff);
		SetVelCutoffApproximate(dptr.gradientVelCutoff);
		SetParticleRadius(dptr.particleRadius);
		SetTerminalSpeed(dptr.rb_terminalSpeed);
		SetRigidBodyCoefficients(dptr.rb_spring, dptr.rb_spring_boundary, dptr.rb_damping);
		SetSurfaceTensionAdhesionCoefficients(dptr.surface_tension_gamma, dptr.surface_adhesion_beta);
		SetGridResolution(dptr.grid_resolution);
		SetBlockSize(dptr.block_size);
		SetGridDimensions(dptr.dimX, dptr.dimY, dptr.dimZ);
		SetLutSize(dptr.lutSize);
		SetKernelSelf(dptr.kernelSelf);
		SetInitialMass(dptr.initialMass);
		SetFluidRestDensity(dptr.fluidRestDensity);
		SetFluidRestVolume(dptr.initialMass, dptr.fluidRestDensity);
		SetZindexStartingVec(dptr.zindexStartingVec);
		SetGamma(dptr.gamma);
		SetFluidGasConstant(dptr.fluidGasConstant, dptr.fluidGasConstantWCSPH);
		SetDensityErrorFactorPCISPH(dptr.param_density_error_factor);
		SetBoundaryConstants(dptr.forceDistance, dptr.maxBoundaryForce, dptr.minCollisionBox, dptr.maxCollisionBox, dptr.addBoundaryForce);
		SetBoundingBox(dptr.minBoundingBox, dptr.maxBoundingBox);
		SetRealContainerValues(dptr.boxLength, dptr.boxHeigth, dptr.boxWidth, dptr.minContainerBox, dptr.maxContainerBox);
		SetDeltaT(dptr.deltaT, dptr.deltaTWCSPH);
		SetScales(dptr.scales);
		SetGravityConstant(dptr.gravityConstant);
		SetInitialMassIntoGravityRatio(dptr.initialMass, dptr.gravityConstant);
		SetFluidViscosityConstant(dptr.fluidViscosityConstant);
		SetSpacing(dptr.spacing);
		SetParticleRenderingSize(dptr.particleRenderingSize);
		SetMaximumArrayLength(dptr.maxLength);

		if(dptr.demoScene == 2)
		{
			SetPipePoints(dptr.pipePoint1, dptr.pipePoint2);
			SetPipeRadius(dptr.pipeRadius);
			SetWallConstant(dptr.wallX);
			SetFluidViscosityConstantTube(dptr.fluidViscosityConstant_tube);
			/*std::cout << dptr.pipePoint1.x << " " << dptr.pipePoint1.y << " " << dptr.pipePoint1.z << std::endl;
			std::cout << dptr.pipePoint2.x << " " << dptr.pipePoint2.y << " " << dptr.pipePoint2.z << std::endl;
			std::cout << dptr.pipeRadius << std::endl;
			std::cout << dptr.wallX << std::endl;*/
		}
		SetOtherConstants();

		/*std::cout << dptr.globalSupportRadius << " " << dptr.kernelSelf << " " << dptr.initialMass << " " << 
		dptr.fluidRestDensity << " " << dptr.fluidGasConstant << " " << dptr.forceDistance << " " << 
		dptr.initialMass << " " << dptr.fluidViscosityConstant << " " << std::endl;*/
	}

	void CreateZIndexTexture( unsigned int* A )
	{	
		CUDA_SAFE_CALL( cudaMallocArray( &array_zindex, &uint_to_tex, 3, 1024 ) );
		CUDA_SAFE_CALL( cudaMemcpyToArray( array_zindex, 0, 0,
			A, 3*1024*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	}

	void CreateLutKernelM4Texture( float* A, const int lutSize )
	{
		CreateLutKernelM4( A, lutSize );
	}

	void CreateLutKernelPressureGradientTexture( float* A, const int lutSize )
	{
		CreateLutKernelPressureGradient( A, lutSize );
	}

	void CreateLutKernelViscosityLapTexture( float* A, const int lutSize )
	{
		CreateLutKernelViscosityLap( A, lutSize );
	}

	void CreateLutSplineSurfaceTensionTexture( float* A, const int lutSize )
	{
		CreateLutSplineSurfaceTension( A, lutSize );
	}

	void CreateLutSplineSurfaceAdhesionTexture( float* A, const int lutSize )
	{
		CreateLutSplineSurfaceAdhesion( A, lutSize );
	}

	void FreeDeviceArrays(dataPointers& dptr)
	{
		if(dptr.tmpArray)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.tmpArray) );
		if(dptr.d_max_predicted_density_value)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_max_predicted_density_value) );	
		if(dptr.d_max_predicted_density_array)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_max_predicted_density_array) );
		if(dptr.d_predictedDensity)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_predictedDensity) );
		if(dptr.d_previous_predicted_density)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_previous_predicted_density) );
		if(dptr.d_densityError)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_densityError) );
		if(dptr.d_correctionPressure)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_correctionPressure) );
		if(dptr.d_previous_correctionPressure)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_previous_correctionPressure) );
		if(dptr.d_pressure)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_pressure) );
		if(dptr.d_density)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_density) );
		if(dptr.d_weighted_volume)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_weighted_volume) );
		if (dptr.d_type[0])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_type[0]) );
		if (dptr.d_type[1])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_type[1]) );
		if (dptr.d_active_type[0])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_active_type[0]) );
		if (dptr.d_active_type[1])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_active_type[1]) );
		if (dptr.d_parent_rb[0])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_parent_rb[0]) );
		if (dptr.d_parent_rb[1])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_parent_rb[1]) );
		if (dptr.d_order_in_child_particles_array[0])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_order_in_child_particles_array[0]) );
		if (dptr.d_order_in_child_particles_array[1])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_order_in_child_particles_array[1]) );	
		if ( dptr.d_force)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_force ) );
		if ( dptr.d_correctionPressureForce)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_correctionPressureForce ) );
		if ( dptr.d_static_force)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_static_force ) );
		if ( dptr.d_dynamic_boundary_force)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_dynamic_boundary_force ) );
		if ( dptr.d_predictedVel)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_predictedVel ) );
		if ( dptr.d_predictedPos)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_predictedPos ) );
		if( dptr.d_smoothcolor_normal )
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_smoothcolor_normal ) );
		if( dptr.d_vel[0] )
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_vel[0] ) );
		if( dptr.d_vel[1] )
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_vel[1] ) );
		if( dptr.d_filtered_pos_zindex )	
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_filtered_pos_zindex ) );

#ifndef USE_VBO_CUDA
		if( dptr.d_pos_zindex[0] )	
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_pos_zindex[0] ) );
		if( dptr.d_pos_zindex[1] )	
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_pos_zindex[1] ) );
#endif

		if( dptr.particlesKVPair_d[0] )
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.particlesKVPair_d[0] ) );
		if( dptr.particlesKVPair_d[1] )	
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.particlesKVPair_d[1] ) );

		if( array_zindex )
			CUDA_SAFE_CALL( cudaFreeArray(array_zindex) );
		
		if ( dptr.d_rigid_bodies)
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_rigid_bodies ) );

		if ( dptr.d_rigid_particle_relative_pos[0])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_rigid_particle_relative_pos[0] ) );
		if ( dptr.d_rigid_particle_relative_pos[1])
			CUDA_SAFE_CALL( cudaFree( (void**)&dptr.d_rigid_particle_relative_pos[1] ) );
	}

	// A one-time copying routine during initialization 
	void CopyZIndicesKVPairHostToDevice(dataPointers& dptr, std::vector<UnifiedParticle>& particles_h)
	{
		const uint countRounded = dptr.particleCountRounded;

		// const uint numParticlesHost = 2 * particles_h.size();
		// uint* kvpairs_h = new uint[numParticlesHost];

		const uint numParticlesHost = 2 * countRounded;
		uint* kvpairs_h = new uint[numParticlesHost];

		uint i = 0;
		for(i = 0; i < particles_h.size(); i++ )
		{
			kvpairs_h[2*i] = particles_h[i].index_;
			kvpairs_h[2*i+1] = i;
		}

		// Meant to make array size a multiple of block size of CUDA
		for(; i < countRounded; i++ )
		{
			kvpairs_h[2*i] = std::numeric_limits<unsigned int>::max();
			kvpairs_h[2*i+1] = i;
		}

		/*
		CUDA_SAFE_CALL( cudaMemcpy( dptr.particlesKVPair_d[0], kvpairs_h, 
		2 * particles_h.size() * sizeof(uint), cudaMemcpyHostToDevice ) );
		*/

		CUDA_SAFE_CALL( cudaMemcpy( dptr.particlesKVPair_d[0], kvpairs_h, 
			2 * countRounded * sizeof(uint), cudaMemcpyHostToDevice ) );

		delete[] kvpairs_h;		
		kvpairs_h = NULL;
	}

	void CopyIndicesDeviceToHost(dataPointers& dptr, uint*& indices)
	{
		CUDA_SAFE_CALL( cudaMemcpy( indices, dptr.particlesKVPair_d[1],
			2 * dptr.particleCount * sizeof(uint), cudaMemcpyDeviceToHost ) );
	}

	void CopyParticleDataHostToDevice(dataPointers& dptr, std::vector<UnifiedParticle>& particles_h)
	{
		int pcountLocal					= dptr.particleCountRounded;

#ifndef USE_VBO_CUDA
		float* pos						= (float*)malloc(4*pcountLocal*sizeof(float));
#endif

		float* velocity					= (float*)malloc(4*pcountLocal*sizeof(float));
		float* pressure					= (float*)malloc(pcountLocal*sizeof(float));
		float* density					= (float*)malloc(pcountLocal*sizeof(float));
		float* magnitude_gradient_color	= (float*)malloc(pcountLocal*sizeof(float));
		float* weighted_volume			= (float*)malloc(pcountLocal*sizeof(float));
		int*   type						= (int*)malloc(pcountLocal*sizeof(int));
		int*   active_type				= (int*)malloc(pcountLocal*sizeof(int));	// 0: active  1: semi-active   2: passive 100:initial value
		int*   parent_rb				= (int*)malloc(pcountLocal*sizeof(int));
		int*   order_in_child_particles = (int*)malloc(pcountLocal*sizeof(int));
		float* corr_press_force			= (float*)malloc(4*pcountLocal*sizeof(float));
		float* corr_pressure			= (float*)malloc(pcountLocal*sizeof(float));
		float* force					= (float*)malloc(4*pcountLocal*sizeof(float));
		float* relative_pos				= (float*)malloc(4*pcountLocal*sizeof(float));

		int i = 0;
		for(i = 0; i < particles_h.size(); i++ )
		{	

#ifndef USE_VBO_CUDA

			pos[4*i]					= particles_h[i].position.x;
			pos[4*i+1]					= particles_h[i].position.y;
			pos[4*i+2]					= particles_h[i].position.z;
			pos[4*i+3]					= particles_h[i].index;

#endif

			velocity[4*i]				= particles_h[i].velocity_.x;
			velocity[4*i+1]				= particles_h[i].velocity_.y;
			velocity[4*i+2]				= particles_h[i].velocity_.z;
			velocity[4*i+3]				= 0.0f;

			pressure[i]					= particles_h[i].pressure_;
			density[i]					= particles_h[i].density_;
			magnitude_gradient_color[i] = 0.0f;
			weighted_volume[i]			= particles_h[i].weighted_volume_;
			type[i]						= particles_h[i].type_;
			active_type[i]				= 100;
			parent_rb[i]				= particles_h[i].parent_rigid_body_index_;
			order_in_child_particles[i] = particles_h[i].order_in_child_particles_array_;

			corr_press_force[4*i]		= 0.0f;
			corr_press_force[4*i+1]		= 0.0f;
			corr_press_force[4*i+2]		= 0.0f;
			corr_press_force[4*i+3]		= 0.0f;

			corr_pressure[i]			= 0.0f;

			relative_pos[4*i]			= particles_h[i].init_relative_pos_.x;
			relative_pos[4*i+1]			= particles_h[i].init_relative_pos_.y;
			relative_pos[4*i+2]			= particles_h[i].init_relative_pos_.z;
			relative_pos[4*i+3]			= 0.0f;

			force[4*i]					= 0.0f;
			force[4*i+1]				= 0.0f;
			force[4*i+2]				= 0.0f;
			force[4*i+3]				= 0.0f;
		}

		// Meant to be zero particles to make array size a multiple of 
		// block size of CUDA
		for( ; i < dptr.particleCountRounded; i++ )
		{
#ifndef USE_VBO_CUDA

			pos[4*i]					= dptr.maxCollisionBox.x * 4.0;
			pos[4*i+1]					= dptr.maxCollisionBox.y * 4.0;
			pos[4*i+2]					= dptr.maxCollisionBox.z * 4.0;
			pos[4*i+3]					= std::numeric_limits<unsigned int>::max();

#endif

			velocity[4*i]				= 0.0;
			velocity[4*i+1]				= 0.0;
			velocity[4*i+2]				= 0.0;
			velocity[4*i+3]				= 0.0;

			pressure[i]					= 0.0;
			density[i]					= 0.0;
			magnitude_gradient_color[i] = 0.0;
			weighted_volume[i]			= fc->fluidRestVol;
			type[i]						= 100;		// not a useful particle
			active_type[i]				= 100;		
			parent_rb[i]				= -1;	// not a useful rigid body index
			order_in_child_particles[i] = -1; // not a useful index

			corr_press_force[4*i]		= 0.0f;
			corr_press_force[4*i+1]		= 0.0f;
			corr_press_force[4*i+2]		= 0.0f;
			corr_press_force[4*i+3]		= 0.0f;

			corr_pressure[i]			= 0.0f;

			relative_pos[4*i]			= dptr.maxCollisionBox.x * 4.0;
			relative_pos[4*i+1]			= dptr.maxCollisionBox.x * 4.0;
			relative_pos[4*i+2]			= dptr.maxCollisionBox.x * 4.0;
			relative_pos[4*i+3]			= 0.0f;

			force[4*i]					= 0.0f;
			force[4*i+1]				= 0.0f;
			force[4*i+2]				= 0.0f;
			force[4*i+3]				= 0.0f;
		}

#ifndef USE_VBO_CUDA

		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_pos_zindex[0], pos, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );		

#endif

		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_vel[0], velocity, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_pressure, pressure, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );	
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_density, density, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_predictedDensity, density, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_previous_predicted_density, density, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_weighted_volume, weighted_volume, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_type[0], type, 
			dptr.particleCountRounded * sizeof(int), cudaMemcpyHostToDevice ) );	
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_active_type[0], active_type, 
			dptr.particleCountRounded * sizeof(int), cudaMemcpyHostToDevice ) );	
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_parent_rb[0], parent_rb, 
			dptr.particleCountRounded * sizeof(int), cudaMemcpyHostToDevice ) );		
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_order_in_child_particles_array[0], order_in_child_particles, 
			dptr.particleCountRounded * sizeof(int), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_force, force, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_static_force, force, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_dynamic_boundary_force, force, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_correctionPressureForce, corr_press_force, 
			dptr.particleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_correctionPressure, corr_pressure, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_previous_correctionPressure, corr_pressure, 
			dptr.particleCountRounded * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_rigid_particle_relative_pos[0], relative_pos, 
			dptr.particleCountRounded *sizeof(float4), cudaMemcpyHostToDevice ) );

#ifndef USE_VBO_CUDA
		free(pos);
#endif

		free(velocity);
		free(pressure);
		free(density);
		free(weighted_volume);
		free(type);
		free(parent_rb);
		free(order_in_child_particles);
		free(force);
		free(corr_press_force);
		free(corr_pressure);
		free(relative_pos);
		
	}

	void CopyRigidBodyDataHostToDevice(const std::vector<RigidBody*>& rigidbodies_h, dataPointers& dptr)
	{
		const int num_rigid_bodies = rigidbodies_h.size();
		RigidBody_GPU* h_rigid_bodies = (RigidBody_GPU*)malloc(num_rigid_bodies*sizeof(RigidBody_GPU));	

		for (int i = 0; i < num_rigid_bodies; ++i)
		{
			RigidBody* rb = rigidbodies_h[i];
			if (rb)
			{
				h_rigid_bodies[i].rotation_matrix.m_row[0].x = rb->rotation_matrix().elements[0][0];
				h_rigid_bodies[i].rotation_matrix.m_row[0].y = rb->rotation_matrix().elements[0][1];
				h_rigid_bodies[i].rotation_matrix.m_row[0].z = rb->rotation_matrix().elements[0][2];
				h_rigid_bodies[i].rotation_matrix.m_row[0].w = 0.0f;
				h_rigid_bodies[i].rotation_matrix.m_row[1].x = rb->rotation_matrix().elements[1][0];
				h_rigid_bodies[i].rotation_matrix.m_row[1].y = rb->rotation_matrix().elements[1][1];
				h_rigid_bodies[i].rotation_matrix.m_row[1].z = rb->rotation_matrix().elements[1][2];
				h_rigid_bodies[i].rotation_matrix.m_row[1].w = 0.0f;
				h_rigid_bodies[i].rotation_matrix.m_row[2].x = rb->rotation_matrix().elements[2][0];
				h_rigid_bodies[i].rotation_matrix.m_row[2].y = rb->rotation_matrix().elements[2][1];
				h_rigid_bodies[i].rotation_matrix.m_row[2].z = rb->rotation_matrix().elements[2][2];
				h_rigid_bodies[i].rotation_matrix.m_row[2].w = 0.0f;

				h_rigid_bodies[i].inv_inertia_local.m_row[0].x = rb->inverted_inertia_local().elements[0][0];
				h_rigid_bodies[i].inv_inertia_local.m_row[0].y = rb->inverted_inertia_local().elements[0][1];
				h_rigid_bodies[i].inv_inertia_local.m_row[0].z = rb->inverted_inertia_local().elements[0][2];
				h_rigid_bodies[i].inv_inertia_local.m_row[0].w = 0.0f;
				h_rigid_bodies[i].inv_inertia_local.m_row[1].x = rb->inverted_inertia_local().elements[1][0];
				h_rigid_bodies[i].inv_inertia_local.m_row[1].y = rb->inverted_inertia_local().elements[1][1];
				h_rigid_bodies[i].inv_inertia_local.m_row[1].z = rb->inverted_inertia_local().elements[1][2];
				h_rigid_bodies[i].inv_inertia_local.m_row[1].w = 0.0f;
				h_rigid_bodies[i].inv_inertia_local.m_row[2].x = rb->inverted_inertia_local().elements[2][0];
				h_rigid_bodies[i].inv_inertia_local.m_row[2].y = rb->inverted_inertia_local().elements[2][1];
				h_rigid_bodies[i].inv_inertia_local.m_row[2].z = rb->inverted_inertia_local().elements[2][2];
				h_rigid_bodies[i].inv_inertia_local.m_row[2].w = 0.0f;

				h_rigid_bodies[i].inv_inertia_world.m_row[0].x = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[0].y = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[0].z = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[0].w = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[1].x = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[1].y = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[1].z = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[1].w = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[2].x = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[2].y = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[2].z = 0.0f;
				h_rigid_bodies[i].inv_inertia_world.m_row[2].w = 0.0f;

				h_rigid_bodies[i].pos				= make_float4(rb->rigidbody_pos().x, rb->rigidbody_pos().y, rb->rigidbody_pos().z, 0.0f);
				h_rigid_bodies[i].vel				= make_float4(rb->velocity().x, rb->velocity().y, rb->velocity().z, 0.0f);
				h_rigid_bodies[i].angular_velocity  = make_float4(rb->angular_velocity().x, rb->angular_velocity().y, rb->angular_velocity().z, 0.0f);
				h_rigid_bodies[i].force				= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_rigid_bodies[i].torque			= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_rigid_bodies[i].linear_momentum	= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_rigid_bodies[i].angular_momentum	= make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_rigid_bodies[i].quaternion		= make_float4(rb->quaternion().x, rb->quaternion().y, rb->quaternion().z, 0.0f);
				h_rigid_bodies[i].replacement		= make_float4(0.0f, 0.0f, 0.0f, 0.0f);

				// set rigid_particle_indices
				const std::vector<int>& rp_indices = rb->rigid_particle_indices();
				const int num_particles = rp_indices.size();
				for (int j = 0; j < num_particles; ++j)
				{
					h_rigid_bodies[i].rigid_particle_indices[j] = rp_indices[j];
				}				
				for (int j = num_particles; j < MAX_NUM_RIGID_PARTICLES; ++j)
				{
					h_rigid_bodies[i].rigid_particle_indices[j] = -1; // set others as incorrect value
				}	

				h_rigid_bodies[i].mass				= rb->mass();
				h_rigid_bodies[i].num_particles		= rb->rigid_particle_indices().size();		
			}	
		}

		// copy to device mem
		CUDA_SAFE_CALL( cudaMemcpy(dptr.d_rigid_bodies, h_rigid_bodies, num_rigid_bodies*sizeof(RigidBody_GPU), cudaMemcpyHostToDevice) );
		
		// clean up
		if (h_rigid_bodies)
			free(h_rigid_bodies);

	}

	/*
	void CopyAllParticleLayersHostToDevice( dataPointers& dptr, float* pos_zindex_Host, float* vel_pressure_Host, unsigned int* zindex_Host, 
		uint count )
	{
#ifndef USE_VBO_CUDA
		float* pos_zindex = (float*)malloc( 4 * dptr.finalParticleCountRounded * sizeof(float) );
#endif

		float* vel_pressure = (float*)malloc( 4 * dptr.finalParticleCountRounded * sizeof(float) );
		uint* zindex = (uint*)malloc( 2 * dptr.finalParticleCountRounded * sizeof(uint) );


		int lcount = 0;
		while( lcount + count < dptr.finalParticleCount)
		{
			for( int i = 0; i < count; i++ )
			{
#ifndef USE_VBO_CUDA
				pos_zindex[4*(lcount+i)] = pos_zindex_Host[4*i];
				pos_zindex[4*(lcount+i)+1] = pos_zindex_Host[4*i+1];
				pos_zindex[4*(lcount+i)+2] = pos_zindex_Host[4*i+2];
				pos_zindex[4*(lcount+i)+3] = pos_zindex_Host[4*i+3];
#endif

				vel_pressure[4*(lcount+i)] = vel_pressure_Host[4*i];
				vel_pressure[4*(lcount+i)+1] = vel_pressure_Host[4*i+1];
				vel_pressure[4*(lcount+i)+2] = vel_pressure_Host[4*i+2];
				vel_pressure[4*(lcount+i)+3] = vel_pressure_Host[4*i+3];

				//std::cout << pos_zindex_Host[4*i] << " " << pos_zindex_Host[4*i+1] << " " <<
				//pos_zindex_Host[4*i+2] << " " << pos_zindex_Host[4*i+3] << " " << lcount+i << std::endl;

				zindex[2*(lcount+i)] = zindex_Host[2*i];
				zindex[2*(lcount+i)+1] = lcount+i;
			}
			lcount += count;
		}

		while(lcount < dptr.finalParticleCountRounded)
		{
#ifndef USE_VBO_CUDA
			pos_zindex[4*lcount] = dptr.maxCollisionBox.x * 4.0;;
			pos_zindex[4*lcount+1] = dptr.maxCollisionBox.y * 4.0;
			pos_zindex[4*lcount+2] = dptr.maxCollisionBox.z * 4.0;;
			pos_zindex[4*lcount+3] = std::numeric_limits<unsigned int>::max();
#endif

			zindex[2*lcount] = std::numeric_limits<unsigned int>::max();
			zindex[2*lcount+1] = lcount;

			vel_pressure[4*lcount] = 0.0;
			vel_pressure[4*lcount+1] = 0.0;
			vel_pressure[4*lcount+2] = 0.0;
			vel_pressure[4*lcount+3] = 0.0;

			lcount++;
		}


		//std::cout << lcount << " **" << " " << dptr.finalParticleCount << " " << dptr.finalParticleCountRounded <<  std::endl;

		dptr.particleCount = 0;
		dptr.finalParticleCount = lcount;


		CUDA_SAFE_CALL( cudaMemcpy( dptr.particlesKVPair_d[0], zindex, 
			2 * dptr.finalParticleCountRounded * sizeof(uint), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.particlesKVPair_d[1], zindex, 
			2 * dptr.finalParticleCountRounded * sizeof(uint), cudaMemcpyHostToDevice ) );

#ifndef USE_VBO_CUDA

		CUDA_SAFE_CALL( cudaMemcpy( dptr.array_pos_zindex[0], pos_zindex, 
			dptr.finalParticleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );

		CUDA_SAFE_CALL( cudaMemcpy( dptr.array_pos_zindex[1], pos_zindex, 
			dptr.finalParticleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );

#endif

		CUDA_SAFE_CALL( cudaMemcpy( dptr.array_vel_pressure[0], vel_pressure, 
			dptr.finalParticleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.array_vel_pressure[1], vel_pressure, 
			dptr.finalParticleCountRounded * sizeof(float4), cudaMemcpyHostToDevice ) );

#ifndef USE_VBO_CUDA
		free(pos_zindex);
#endif

		free(vel_pressure);
		free(zindex);
	}
	*/

	// This function allows to add more particles on to GPU from host per frame

	/*
	void CopyParticleLayerHostToDevice(dataPointers& dptr, uint count)
	{
		if( dptr.particleCount + count > dptr.finalParticleCount )
			return;

		uint countRounded = dptr.particleCount + count;
		if( countRounded % MAX_THREADS_PER_BLOCK_SPH )
			countRounded += ( MAX_THREADS_PER_BLOCK_SPH - countRounded % MAX_THREADS_PER_BLOCK_SPH );
		dptr.particleCountRounded = countRounded;

		dptr.particleCount += count;

		//SetDeviceConstants(dptr);

		SetParticleCount(dptr.particleCount);
		//float rad = 0.0166667;
		//SetGlobalSupportRadius(rad/2.0);

		//std::cout << dptr.particleCount << " " << count << " " << dptr.finalParticleCount <<  " " << dptr.particleCountRounded << std::endl;
	}
	*/

	/*
	void CopyParticleLayerHostToDevice2(dataPointers& dptr, uint count, float* pos_zindex_Host, float* vel_pressure_Host, unsigned int* zindex_Host)
	{
		if( dptr.particleCount + count > dptr.finalParticleCount )
			return;


		for( int i = 0; i < count; i++ )
		{
			zindex_Host[2*i+1] = dptr.particleCount+i;
		}

		CUDA_SAFE_CALL( cudaMemcpy( &dptr.particlesKVPair_d[0][dptr.particleCount], zindex_Host, 
			2 * count * sizeof(uint), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( &dptr.particlesKVPair_d[1][dptr.particleCount], zindex_Host, 
			2 * count * sizeof(uint), cudaMemcpyHostToDevice ) );

#ifndef USE_VBO_CUDA
		CUDA_SAFE_CALL( cudaMemcpy( &dptr.array_pos_zindex[0][dptr.particleCount], pos_zindex_Host, 
			count * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( &dptr.array_pos_zindex[1][dptr.particleCount], pos_zindex_Host, 
			count * sizeof(float4), cudaMemcpyHostToDevice ) );
#endif

		CUDA_SAFE_CALL( cudaMemcpy( &dptr.array_vel_pressure[0][dptr.particleCount], vel_pressure_Host, 
			count * sizeof(float4), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( &dptr.array_vel_pressure[1][dptr.particleCount], vel_pressure_Host, 
			count * sizeof(float4), cudaMemcpyHostToDevice ) )


			uint countRounded = dptr.particleCount + count;
		if( countRounded % MAX_THREADS_PER_BLOCK_SPH )
			countRounded += ( MAX_THREADS_PER_BLOCK_SPH - countRounded % MAX_THREADS_PER_BLOCK_SPH );
		dptr.particleCountRounded = countRounded;

		dptr.particleCount += count;

		SetParticleCount(dptr.particleCount);
	}
	*/

	void CopyParticlesDeviceToHost(dataPointers& dptr, std::vector<UnifiedParticle>& particles_h, float*& p)
	{	
		CUDA_SAFE_CALL( cudaMemcpy( &p[0], dptr.d_pos_zindex[dptr.currIndex], 
			dptr.particleCount * sizeof(float4), cudaMemcpyDeviceToHost ) );
	}

#ifdef OUTPUT_GPU_RIGID_BODY_INFO

	void UpdateRigidBodyPovInfo(dataPointers& dptr, vector<RigidBodyOfflineRenderingInfo>& rigid_body_info_)
	{
		// TODO: Currently we simply copy all rigid body information back to host, this could be optimized
		const size_t num_rigid_bodies = dptr.num_rigid_bodies;
		RigidBody_GPU* h_rigid_bodies = (RigidBody_GPU*)malloc(num_rigid_bodies*sizeof(RigidBody_GPU));	

		// copy to host mem
		CUDA_SAFE_CALL( cudaMemcpy(h_rigid_bodies, dptr.d_rigid_bodies, num_rigid_bodies*sizeof(RigidBody_GPU), cudaMemcpyDeviceToHost) );

		for (int i = 0; i < num_rigid_bodies; ++i)
		{
			Vector3f new_pos(h_rigid_bodies[i].pos.x, h_rigid_bodies[i].pos.y, h_rigid_bodies[i].pos.z);
			Matrix3x3 new_rotation_matrix;
			new_rotation_matrix.elements[0][0] = h_rigid_bodies[i].rotation_matrix.m_row[0].x;
			new_rotation_matrix.elements[0][1] = h_rigid_bodies[i].rotation_matrix.m_row[0].y;
			new_rotation_matrix.elements[0][2] = h_rigid_bodies[i].rotation_matrix.m_row[0].z;
			new_rotation_matrix.elements[1][0] = h_rigid_bodies[i].rotation_matrix.m_row[1].x;
			new_rotation_matrix.elements[1][1] = h_rigid_bodies[i].rotation_matrix.m_row[1].y;
			new_rotation_matrix.elements[1][2] = h_rigid_bodies[i].rotation_matrix.m_row[1].z;
			new_rotation_matrix.elements[2][0] = h_rigid_bodies[i].rotation_matrix.m_row[2].x;
			new_rotation_matrix.elements[2][1] = h_rigid_bodies[i].rotation_matrix.m_row[2].y;
			new_rotation_matrix.elements[2][2] = h_rigid_bodies[i].rotation_matrix.m_row[2].z;
			rigid_body_info_[i].updateVerticesAndNormals(new_pos, new_rotation_matrix);
		}

		// clean up
		if (h_rigid_bodies)
			free(h_rigid_bodies);
	}

#endif

	// SPH Methods
	void ZindexSorting(int currIndex, int currType, dataPointers& dptr)
	{
		// Sort the particle key-value pair(zindex, i) using radix-sort
		// dptr.particlesKVPair_d[0] is both the input and output array - data will be sorted
		// dptr.particlesKVPair_d[1] is just an additional array to allow ping pong computation
		RadixSort( (KeyValuePair*)dptr.particlesKVPair_d[0], (KeyValuePair*)dptr.particlesKVPair_d[1], dptr.particleCount, 32 );
		// Use the old attributes array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],								dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[currType],								dptr.particleCountRounded*sizeof(int) ) );	

		CUDA_SAFE_CALL( cudaBindTextureToArray( texture_zindex_array, array_zindex ) );

		// Copy sorted particles values to the other ping pong array
		CopySortedParticleValuesKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(		dptr.particleCountRounded, 
																						dptr.d_pos_zindex[1-currIndex], 																						
																						dptr.d_vel[1-currIndex], 
																						dptr.d_type[1-currType], 
																						dptr.particlesKVPair_d[0] );

		// Use the newly copied array to bind as texture
		// Now the sorted attributes arrays are stored in [1-currIndex] buffers
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[1-currType],							dptr.particleCountRounded*sizeof(int) ) );	
	}

	void ZindexSortingApproximate(int currIndex, int currActiveType, dataPointers& dptr)
	{
		// Sort the particle key-value pair(zindex, i) using radix-sort
		// dptr.particlesKVPair_d[0] is both the input and output array - data will be sorted
		// dptr.particlesKVPair_d[1] is just an additional array to allow ping pong computation
		RadixSort( (KeyValuePair*)dptr.particlesKVPair_d[0], (KeyValuePair*)dptr.particlesKVPair_d[1], 
			dptr.particleCount, 32 );
		// Use the old attributes array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],								dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_active_type,						dptr.d_active_type[currActiveType],					dptr.particleCountRounded*sizeof(int) ) );	

		CUDA_SAFE_CALL( cudaBindTextureToArray( texture_zindex_array, array_zindex ) );

		// Copy sorted particles values to the other ping pong array
		CopySortedParticleValuesKernelApproximate<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particleCountRounded, 
			dptr.d_pos_zindex[1-currIndex], 																						
			dptr.d_vel[1-currIndex], 
			dptr.d_active_type[1-currActiveType], 
			dptr.particlesKVPair_d[0] );

		// Use the newly copied array to bind as texture
		// Now the sorted attributes arrays are stored in [1-currIndex] buffers
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_active_type,						dptr.d_active_type[1-currActiveType],				dptr.particleCountRounded*sizeof(int) ) );	
	}

	void BlockGeneration(dataPointers& dptr, int& x, int& y)
	{
		// Number of CUDA blocks
		// x : without redistribution
		// y : after redistribution

		CUDA_SAFE_CALL( cudaMemcpyToSymbol(blockCount, &x, sizeof(uint)) );

		// Clear index arrays
		// Here we use part of dptr.particlesKVPair_d[1] as an ancillary space to store new grid pairs(i.e. (grid_start_index, num_particles)) 
		// num_particles: Count of particles in this block, grid_start_index: Starting index of particles within this block 
		ClearIndexArrays<<< gridsize_indices, MAX_THREADS_PER_BLOCK_SPH >>>(dptr.particlesKVPair_d[1]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: ClearIndexArrays: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		CalculateBlocksKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0], dptr.tmpArray);

		cudaError_t error2 = cudaGetLastError();
		if (error2 != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateBlocksKernel: %s\n", cudaGetErrorString(error2) );
		}    
		cudaDeviceSynchronize ();

		CUDA_SAFE_CALL( cudaMemcpyFromSymbol(&x, blockCount, sizeof(uint)) );

		dptr.filledBlocks = x;

		/*==============================================================================================*/
		/*
		int ii;
		for( int i = 0; i < dptr.particleCount; i++ )
		{
		cudaMemcpy(&ii, &dptr.tmpArray[i], sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << ii << " ";
		//cudaMemcpy(&ii, &dptr.particlesKVPair_d[0][2*i+1], sizeof(int), cudaMemcpyDeviceToHost);
		//   std::cout << ii << " * ";
		}
		std::cout << std::endl << std::endl;
		//*/

#ifdef DUMP_PARTICLES_TO_FILE

		CUDA_SAFE_CALL( cudaMemcpy( dptr.cpuMem1, dptr.particlesKVPair_d[1], 2 * dptr.dimX * dptr.dimY * dptr.dimZ * sizeof(uint),
			cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.cpuMem2, dptr.d_pos_zindex[1-dptr.currIndex], dptr.particleCount * sizeof(float4),
			cudaMemcpyDeviceToHost ) );
#endif

		/*==============================================================================================*/

		RedistributeBlocksKernel<<<x, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0]);

		// Now dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)

		CUDA_SAFE_CALL( cudaMemcpyFromSymbol(&y, blockCount, sizeof(uint)) );
	}

	void DensityPressureComputation(int currIndex, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays
		if (fc->physicsType == 'o')
		{
			CalculateDensitiesInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateDensitiesInBlocksKernelWCSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed density and pressure arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pressure, dptr.d_pressure, dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_density, dptr.d_density, dptr.particleCountRounded*sizeof(float) ) );

		dptr.filteredParticleCount = 0;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(filteredCount, &dptr.filteredParticleCount, sizeof(uint)) );
	}

	void DensityPressureComputationApproximate(int currIndex, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays
		if (fc->physicsType == 'o')
		{
			CalculateDensitiesInBlocksKernelSPHApproximate<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateDensitiesInBlocksKernelWCSPHApproximate<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed density and pressure arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pressure, dptr.d_pressure, dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_density, dptr.d_density, dptr.particleCountRounded*sizeof(float) ) );

		dptr.filteredParticleCount = 0;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(filteredCount, &dptr.filteredParticleCount, sizeof(uint)) );
	}

	void FindActiveParticles(int currActiveType, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays

		FindActiveParticlesInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_active_type[1-currActiveType], dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: FindActiveParticlesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();
	}

	void MarkMoreActiveParticlesState(int currActiveType, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays

		MarkMoreActiveParticlesStateInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_active_type[1-currActiveType],
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: MarkMoreActiveParticlesStateInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();
	}

	void MarkAllRemainingParticlesState(int currActiveType, dataPointers& dptr, int& y)
	{
		MarkAllRemainingParticlesStateInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_active_type[1-currActiveType],
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: MarkAllRemainingParticlesStateInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// bind active type texture again
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_active_type,						dptr.d_active_type[1-currActiveType],				dptr.particleCountRounded*sizeof(int) ) );	
	}

	void ForceComputation(int currIndex, dataPointers& dptr, int& y)
	{
		// Previously the updated pressure & density arrays are stored in [currIndex] buffers & the corresponding textures are also binded to these arrays
		// But the pos & vel buffers are still stored in [1-currIndex] buffers
		if (fc->physicsType == 'o')
		{
			CalculateForcesInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateForcesInBlocksKernelWCSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Now all the updated position & velocity arrays are stored in [currIndex] buffers
		// Bind the newly computed position & velocity arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex],	dptr.particleCountRounded*sizeof(float4) ) );

	}

	void ForceComputationApproximate(int currIndex, dataPointers& dptr, int& y)
	{
		// Previously the updated pressure & density arrays are stored in [currIndex] buffers & the corresponding textures are also binded to these arrays
		// But the pos & vel buffers are still stored in [1-currIndex] buffers
		if (fc->physicsType == 'o')
		{
			CalculateForcesInBlocksKernelSPHApproximate<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateForcesInBlocksKernelWCSPHApproximate<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Now all the updated position & velocity arrays are stored in [currIndex] buffers
		// Bind the newly computed position & velocity arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex],	dptr.particleCountRounded*sizeof(float4) ) );
	}

	void ExtractSurface(dataPointers& dptr, int& x)
	{
#ifdef SPH_PROFILING_VERBOSE

		cudaEventRecord(startSurfaceExtraction, 0);

#endif

		ExtractSurfaceParticlesKernel2<<<x, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_filtered_pos_zindex, 
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: ExtractSurfaceParticlesKernel2: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

#ifdef SPH_PROFILING_VERBOSE

		cudaEventRecord(stopSurfaceExtraction, 0);
		cudaEventSynchronize(stopSurfaceExtraction);
		cudaEventElapsedTime(&elapsedTime, startSurfaceExtraction, stopSurfaceExtraction);
		cudaEventDestroy(startSurfaceExtraction);
		cudaEventDestroy(stopSurfaceExtraction);
		surfaceExtractionTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventRecord(startSimulation, 0);

#endif

		CUDA_SAFE_CALL( cudaMemcpyFromSymbol(&dptr.filteredParticleCount, filteredCount, sizeof(uint)) );
		//std::cout << "particles extracted: " << dptr.filteredParticleCount << std::endl;
	}

	void CalculateZindex(int currIndex, dataPointers& dptr)
	{
		// Until Now all the updated attribute arrays except zindex buffers are stored in [currIndex] buffers
		FillZindicesKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pos_zindex[currIndex], dptr.particlesKVPair_d[0]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: FillZindicesKernel: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Finally we make sure all the updated attribute arrays are stored in [currIndex] buffers

	}

	void UnbindTexturesSPH()
	{
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_pos_zindex ) );		
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_vel ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_type ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_zindex_array ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_pressure ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_density ) );
	}

	void UnbindTexturesSPHApproximate()
	{
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_pos_zindex ) );		
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_vel ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_active_type ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_zindex_array ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_pressure ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_density ) );
	}

	void UnbindTextureRigidFluidSPH()
	{
		UnbindTexturesSPH();
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_parent_rb ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_relative_pos) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_order_in_child_particles_array) );
	}

	void UnbindTexturesPCISPH()
	{
		UnbindTexturesSPH();
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_predicted_pos) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_static_force) );
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_corr_pressure_force) );
	}

	void UnbindTextureRigidFluidPCISPH()
	{
		UnbindTexturesPCISPH();
		CUDA_SAFE_CALL( cudaUnbindTexture( texture_dynamic_boundary_force) );
	}

	void PingPongScheme(dataPointers& dptr)
	{
		// Now the updated array_vel_pressure & d_density arrays are stored in [1-currIndex] buffers
		// While the updated array_pos_zindex arrays are stored in [currIndex] buffers
		// So here we copy the array_pos_zindex arrays from [currIndex] buffers to [1-currIndex] buffers

		// Copy sorted particles in the ping pong scheme
		/*
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_pos_zindex[1-currIndex], dptr.d_pos_zindex[currIndex], 
			dptr.particleCount * sizeof(float4), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dptr.d_zindex[1-currIndex], dptr.d_zindex[currIndex], 
			dptr.particleCount * sizeof(float), cudaMemcpyDeviceToDevice ) );
		*/

		// Now all the updated attribute arrays are stored in [1-currIndex] buffers

		// Here we switch the buffers for the next step
		dptr.currType = 1 - dptr.currType;			// for CUDA computing
		//std::swap(dptr.d_rigid_particle_indices[0], dptr.d_rigid_particle_indices[1]); 
	}

	void PingPongSchemeApproximate(dataPointers& dptr)
	{
		// Here we switch the buffers for the next step
		dptr.currActiveType = 1 - dptr.currActiveType;			// for CUDA computing
	}

	void PingPongSchemeRigidBody(dataPointers& dptr)
	{
		// Now all the updated attribute arrays are stored in [1-currIndex] buffers
		// Here we switch the buffers for the next step
		dptr.currIndex = 1 - dptr.currIndex;
	}

	void ParticlePhysicsOnDeviceSPH(dataPointers& dptr)
	{	

#ifdef SPH_PROFILING_VERBOSE

		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventRecord(startSimulation, 0);

#endif

		int currIndex = dptr.currIndex;
		int currType = dptr.currType;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

		// Z-index and Sorting
		ZindexSorting(currIndex, currType, dptr);


		// Block Generation
		BlockGeneration(dptr, x, y);

		CalculateWeightedVolume(currIndex, dptr, y);

		// for each liquid particle, we compute its density & pressure using SPH/WCSPH/PCISPH
		if (fc->addWallWeightFunction)
		{
			CorrectedDensityPressureComputationWallWeight(currIndex, dptr, y);
		}
		else
		{
			CorrectedDensityPressureComputation(currIndex, dptr, y);
		}

		// for each 
		// (1) liquid particle, we calculate its force from liquid particles, rigid particles and boundaries
		// (2) rigid particle, we calculate its force from liquid particles, rigid particles and boundaries 
		ForceComputation(currIndex, dptr, y);
		//ForceComputationVersatileCoupling(currIndex, dptr, y);

		// Extract Surface
		//ExtractSurface(dptr, x);


		// Calculate Z-indices for particles
		CalculateZindex(currIndex, dptr);


		UnbindTexturesSPH();

		// for sph, we don't need to switch the buffers for rendering 
		// Because now all the updated attribute arrays are stored in [currIndex] buffers
		PingPongScheme(dptr);


		// Synchronizing
		cudaDeviceSynchronize();


#ifdef SPH_PROFILING_VERBOSE

		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			float averageElapsed = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			/*
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			*/
		}

#endif

	}

	void ParticlePhysicsOnDeviceSPHApproximate(dataPointers& dptr)
	{	

#ifdef SPH_PROFILING_VERBOSE

		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventRecord(startSimulation, 0);

#endif

		int currIndex = dptr.currIndex;
		int currActiveType = dptr.currActiveType;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

		// Z-index and Sorting
		ZindexSortingApproximate(currIndex, currActiveType, dptr);

		// Block Generation
		BlockGeneration(dptr, x, y);

		//-----------------------------Mark state -----------------------------------------
		// Note: we find all active particles at the end of each simulation step
		// here we need to change particle's state according to approximate criteria
		FindActiveParticles(currActiveType, dptr, y);				
		MarkMoreActiveParticlesState(currActiveType, dptr, y);
		MarkAllRemainingParticlesState(currActiveType, dptr, y);

		DensityPressureComputationApproximate(currIndex, dptr, y);

		ForceComputation(currIndex, dptr, y);

		// Extract Surface
		//ExtractSurface(dptr, x);

		// Calculate Z-indices for particles
		CalculateZindex(currIndex, dptr);

		UnbindTexturesSPHApproximate();

		// for sph, we don't need to switch the buffers for rendering 
		// Because now all the updated attribute arrays are stored in [currIndex] buffers
		PingPongSchemeApproximate(dptr);


		// Synchronizing
		cudaDeviceSynchronize();


#ifdef SPH_PROFILING_VERBOSE

		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			float averageElapsed = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			/*
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			*/
		}

#endif

	}

	void ParticlePhysicsOnDevicePCISPH(dataPointers& dptr)
	{	

#ifdef SPH_PROFILING_VERBOSE
		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEvent_t startParallelSorting, startBlockGeneration, startExternalForceCalculation, startPCILoop, startTimeIntegration, startIndexCalculation,
				    stopParallelSorting,  stopBlockGeneration,  stopExternalForceCalculation,  stopPCILoop,  stopTimeIntegration,  stopIndexCalculation;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventCreate(&startParallelSorting);
		cudaEventCreate(&startBlockGeneration);
		cudaEventCreate(&startExternalForceCalculation);
		cudaEventCreate(&startPCILoop);
		cudaEventCreate(&startTimeIntegration);
		cudaEventCreate(&startIndexCalculation);
		cudaEventCreate(&stopParallelSorting);
		cudaEventCreate(&stopBlockGeneration);
		cudaEventCreate(&stopExternalForceCalculation);
		cudaEventCreate(&stopPCILoop);
		cudaEventCreate(&stopTimeIntegration);
		cudaEventCreate(&stopIndexCalculation);
		cudaEventRecord(startSimulation, 0);
#endif

		int currIndex = dptr.currIndex;
		int currType = dptr.currType;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(startParallelSorting, 0); 
#endif


		// Z-index and Sorting
		ZindexSorting(currIndex, currType, dptr);


#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopParallelSorting, 0); 
		cudaEventSynchronize(stopParallelSorting); 
		cudaEventElapsedTime(&elapsedTime, startParallelSorting, stopParallelSorting); 
		parallelSortingTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(startBlockGeneration, 0); 
#endif


		// Block Generation
		BlockGeneration(dptr, x, y);


#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopBlockGeneration, 0); 
		cudaEventSynchronize(stopBlockGeneration); 
		cudaEventElapsedTime(&elapsedTime, startBlockGeneration, stopBlockGeneration); 
		blockGenerationTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(startExternalForceCalculation, 0); 
#endif


		 // External Force Computation
		 CalculateExternalForcesPcisphCUDA(currIndex, dptr, y);


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(stopExternalForceCalculation, 0);
		 cudaEventSynchronize(stopExternalForceCalculation);
		 cudaEventElapsedTime(&elapsedTime, startExternalForceCalculation, stopExternalForceCalculation);
		 externalForceCalculationTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(startPCILoop, 0);
#endif


		 // prediction correction step
		 PredictionCorrectionStepPcisphCUDA(dptr, y);


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(stopPCILoop, 0);
		 cudaEventSynchronize(stopPCILoop);
		 cudaEventElapsedTime(&elapsedTime, startPCILoop, stopPCILoop);
		 pciLoopTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(startTimeIntegration, 0);
#endif


		 // Time Integration
		 TimeIntegrationPcisphCUDAPureFluid(currIndex, dptr, y);


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(stopTimeIntegration, 0);
		 cudaEventSynchronize(stopTimeIntegration);
		 cudaEventElapsedTime(&elapsedTime, startTimeIntegration, stopTimeIntegration);
		 timeIntegrationTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(startSurfaceExtraction, 0);
#endif


		 // Extract Surface
		 //ExtractSurface(dptr, x);


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(stopSurfaceExtraction, 0);
		 cudaEventSynchronize(stopSurfaceExtraction);
		 cudaEventElapsedTime(&elapsedTime, startSurfaceExtraction, stopSurfaceExtraction);
		 surfaceExtractionTimeCounter_ += elapsedTime / 1000.0f;
#endif


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(startIndexCalculation, 0);
#endif


		 // Calculate Z-indices for particles
		 CalculateZindex(currIndex, dptr);


#ifdef SPH_PROFILING_VERBOSE
		 cudaEventRecord(stopIndexCalculation, 0);
		 cudaEventSynchronize(stopIndexCalculation);
		 cudaEventElapsedTime(&elapsedTime, startIndexCalculation, stopIndexCalculation);
		 indexCalculationTimeCounter_ += elapsedTime / 1000.0f;
#endif

		UnbindTexturesPCISPH();

		PingPongScheme(dptr);

		// Synchronizing
		cudaDeviceSynchronize();

#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);
		cudaEventDestroy(startParallelSorting);
		cudaEventDestroy(stopParallelSorting);
		cudaEventDestroy(startBlockGeneration);
		cudaEventDestroy(stopBlockGeneration);
		cudaEventDestroy(startExternalForceCalculation);
		cudaEventDestroy(stopExternalForceCalculation);
		cudaEventDestroy(startPCILoop);
		cudaEventDestroy(stopPCILoop);
		cudaEventDestroy(startTimeIntegration);
		cudaEventDestroy(stopTimeIntegration);
		cudaEventDestroy(startIndexCalculation);
		cudaEventDestroy(stopIndexCalculation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			// physics simulation
			float averageElapsedPhysicsSimulation = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation                  average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsedPhysicsSimulation << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsedPhysicsSimulation << "fps" << std::endl;

			// parallel sorting
			float averageElapsed = parallelSortingTimeCounter_ / frameCounter_;
			std::cout << "  parallel sorting                    average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			// block generation
			averageElapsed = blockGenerationTimeCounter_ / frameCounter_;
			std::cout << "  block generation                    average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			// external force calculation
			averageElapsed = externalForceCalculationTimeCounter_ / frameCounter_;
			std::cout << "  external force calculation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			// pci loop
			averageElapsed = pciLoopTimeCounter_ / frameCounter_;
			std::cout << "  pci loop                            average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			// time integration
			averageElapsed = timeIntegrationTimeCounter_ / frameCounter_;
			std::cout << "  time integration                    average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			// index calculation
			averageElapsed = indexCalculationTimeCounter_ / frameCounter_;
			std::cout << "  index calculation                   average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 100 * averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;

			/*
			// surface extraction
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << averageElapsed / averageElapsedPhysicsSimulation << "%" << std::endl;
			*/


		}
#endif
	}

	void ParticlePhysicsOnDevicePCISPHIhmsen2010(dataPointers& dptr)
	{	

#ifdef SPH_PROFILING_VERBOSE
		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventRecord(startSimulation, 0);
#endif

		int currIndex = dptr.currIndex;
		int currType = dptr.currType;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

		// Z-index and Sorting
		ZindexSorting(currIndex, currType, dptr);


		// Block Generation
		BlockGeneration(dptr, x, y);


		// External Force Computation
		CalculateExternalForcesPcisphCUDA(currIndex, dptr, y);


		// prediction correction step
		PredictionCorrectionStepPcisphCUDA(dptr, y);


		// Time Integration
		TimeIntegrationPcisphCUDAIhmsen2010Method(currIndex, dptr, y);


		// Extract Surface
		//ExtractSurface(dptr, x);


		// Calculate Z-indices for particles
		CalculateZindex(currIndex, dptr);

		UnbindTexturesPCISPH();

		PingPongScheme(dptr);

		// Synchronizing
		cudaDeviceSynchronize();

#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			float averageElapsed = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;

		}
#endif
	}

	void CalculateExternalForcesPcisphCUDA(int currIndex, dataPointers& dptr, int& y)
	{
		CalculateExternalForcesInBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_static_force, dptr.d_correctionPressureForce, dptr.d_correctionPressure,
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateExternalForcesInBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Bind the newly computed external force textures once per time step
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_static_force,			dptr.d_static_force,			dptr.particleCountRounded*sizeof(float4) ) );
	}

	void CalculateExternalForcesStaticBoundariesPcisphCUDA(int currIndex, dataPointers& dptr, int& y)
	{
#if 1

		CalculateExternalForcesStaticBoundariesInBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_static_force, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateExternalForcesStaticBoundariesInBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();
#else

		// debugging purpose (we use this to test situations with only pressure forces)
		CUDA_SAFE_CALL( cudaMemset(dptr.d_static_force,	0.0f,	dptr.particleCountRounded*sizeof(float)) );
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_static_force,		0.0f, dptr. particleCountRounded*sizeof(float)  ): %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

#endif


		// init some quantities which are going to be used in the prediction step
		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed external force textures once per time step
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_static_force,			dptr.d_static_force,			dptr.particleCountRounded*sizeof(float4) ) );
	}
	
	void CalculateExternalForcesFluidRigidCouplingPcisphCUDA(int currIndex, dataPointers& dptr, int& y)
	{
		CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_static_force, 
																										 dptr.particlesKVPair_d[1], 
																										 dptr.particlesKVPair_d[0]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateExternalForcesRigidFluidCouplingInBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// init some quantities which are going to be used in the prediction step
		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed d_static_force & the d_dynamic_boundary_force (from previous frame or initial state) textures once per time step
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_static_force,			dptr.d_static_force,			dptr. particleCountRounded*sizeof(float4)));
	}

	void CalculateExternalForcesWithoutBoundaryForceFluidRigidCouplingPcisphCUDA(int currIndex, dataPointers& dptr, int& y)
	{
		CalculateExternalForcesWithoutBoundaryForceRigidFluidCouplingInBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_static_force, 
			dptr.particlesKVPair_d[1], 
			dptr.particlesKVPair_d[0]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateExternalForcesWithoutBoundaryForceRigidFluidCouplingInBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// init some quantities which are going to be used in the prediction step
		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressure,		0.0f, dptr. particleCountRounded*sizeof(float)  ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		CUDA_SAFE_CALL( cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ) );
		{
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: cudaMemset(dptr.d_correctionPressureForce,  0.0f, dptr. particleCountRounded*sizeof(float4) ): %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed d_static_force & the d_dynamic_boundary_force (from previous frame or initial state) textures once per time step
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_static_force,			dptr.d_static_force,			dptr. particleCountRounded*sizeof(float4)));
	}

	void UpdateLiquidParticlePosVelPCISPH(int currIndex, dataPointers& dptr)
	{
		UpdateLiquidParticleKernelPCISPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(currIndex, dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr, "CUDA ERROR: UpdateLiquidParticleKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}
		cudaDeviceSynchronize ();

		// Use the newly updated pos & vel array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],					dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
	}

	void PredictionCorrectionStepPcisphCUDA(dataPointers& dptr, int& y)
	{
		bool densityErrorTooLarge = true; 
		int iteration = 0;
		while( (iteration < dptr.param_min_loops) || ((densityErrorTooLarge) && (iteration < dptr.param_max_loops)) )	
		{		
			PredictPositionAndVelocityPcisphCUDA(dptr, y);

			ComputePredictedDensityAndPressurePcisphCUDA(dptr, y);

			ComputeCorrectivePressureForcePcisphCUDA(dptr, y);

			//-------------------- TODO:These parts can be executed parallel to ComputeCorrectivePressureForcePcisphCUDA -------------------- 
			float max_predicted_density = 0.0;
			GetMaxPredictedDensityCUDA(dptr, max_predicted_density);
			float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);	// 100/1000 * maxPredictedDensity - 100; 	
			if(densityErrorInPercent < dptr.param_max_density_error_allowed) 
				densityErrorTooLarge = false; // stop loop
			//-------------------------------------------------------------------------------------------------------------------------------

			iteration++;
		}

		// Bind the newly calculated correctionPressureForce textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL,  texture_corr_pressure_force, dptr.d_correctionPressureForce,	dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_corr_pressure,		 dptr.d_correctionPressure,			dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_predicted_density,   dptr.d_predictedDensity,			dptr.particleCountRounded*sizeof(float) ) );
	}

	void PredictionCorrectionStepPcisphCUDAVersatileCoupling(dataPointers& dptr, int& y)
	{
		bool densityErrorTooLarge = true; 
		int iteration = 0;
		while( (iteration < dptr.param_min_loops) || ((densityErrorTooLarge) && (iteration < dptr.param_max_loops)) )	
		{		
			PredictPositionAndVelocityPcisphCUDA(dptr, y);

			ComputePredictedDensityAndPressurePcisphCUDA(dptr, y);

			ComputeCorrectivePressureBoundaryForcePcisphCUDA(dptr, y);

			//-------------------- TODO:These parts can be executed parallel to ComputeCorrectivePressureForcePcisphCUDA -------------------- 
			float max_predicted_density = 0.0;
			GetMaxPredictedDensityCUDA(dptr, max_predicted_density);
			float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);	// 100/1000 * maxPredictedDensity - 100; 	
			if(densityErrorInPercent < dptr.param_max_density_error_allowed) 
				densityErrorTooLarge = false; // stop loop
			//-------------------------------------------------------------------------------------------------------------------------------

			iteration++;
		}

		// Bind the newly calculated correctionPressureForce textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL,  texture_corr_pressure_force, dptr.d_correctionPressureForce,	dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_corr_pressure,		 dptr.d_correctionPressure,			dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_predicted_density,   dptr.d_predictedDensity,			dptr.particleCountRounded*sizeof(float) ) );
	}

	void PredictPositionAndVelocityPcisphCUDA(dataPointers& dptr, int& y)
	{
		// Bind the newly calculated correctionPressureForce textures before the loop starts
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_corr_pressure_force, dptr.d_correctionPressureForce, dptr.particleCountRounded*sizeof(float4) ) );	
	
		PredictPositionAndVelocityBlocksKernelPCISPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_predictedPos );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: PredictPositionAndVelocityBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Use the newly calculated predicted_pos array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_predicted_pos, dptr.d_predictedPos, dptr.particleCountRounded*sizeof(float4) ) );
	}

	void ComputePredictedDensityAndPressurePcisphCUDA(dataPointers& dptr, int& y)
	{
		if (fc->addWallWeightFunction)
		{
			ComputePredictedDensityAndPressureBlocksKernelPCISPHWallWeight<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_densityError, 
																									dptr.d_predictedDensity,
																									dptr.d_correctionPressure, 
																									dptr.particlesKVPair_d[1], 
																									dptr.particlesKVPair_d[0]);
		}
		else
		{
			ComputePredictedDensityAndPressureBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_densityError, 
																									dptr.d_predictedDensity,
																									dptr.d_correctionPressure, 
																									dptr.particlesKVPair_d[1], 
																									dptr.particlesKVPair_d[0]);
		}

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: ComputePredictedDensityAndPressureBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Use the newly calculated attributes array to bind as textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_density_error,		 dptr.d_densityError,		 dptr.particleCountRounded*sizeof(float4) ) );
		//CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_predicted_density,   dptr.d_predictedDensity,	 dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_corr_pressure,		 dptr.d_correctionPressure,  dptr.particleCountRounded*sizeof(float) ) );

	}

	void ComputeCorrectivePressureForcePcisphCUDA(dataPointers& dptr, int& y)
	{
		if (fc->isTwoWayCoupling)
		{
			ComputeCorrectivePressureForceBlocksKernelPCISPHTwoWayCoupling<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_correctionPressureForce, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: ComputeCorrectivePressureForceBlocksKernelPCISPHTwoWayCoupling: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		} 
		else
		{
			ComputeCorrectivePressureForceBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_correctionPressureForce, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: ComputeCorrectivePressureForceBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
	}

	void ComputeCorrectivePressureBoundaryForcePcisphCUDA(dataPointers& dptr, int& y)
	{
		if (fc->isTwoWayCoupling)
		{
			ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPHTwoWayCoupling<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_correctionPressureForce, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: ComputeCorrectivePressureForceBlocksKernelPCISPHTwoWayCoupling: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		} 
		else
		{
			ComputeCorrectivePressureBoundaryForceBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_correctionPressureForce, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: ComputeCorrectivePressureForceBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
	}

	void TimeIntegrationPcisphCUDAPureFluid(int currIndex, dataPointers& dptr, int& y)
	{
		TimeIntegrationBlocksKernelPCISPHPureFluid<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex], 
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: TimeIntegrationBlocksKernelPCISPHPureFluid: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Use the newly calculated pos_zindex array to bind as textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex], dptr.particleCountRounded*sizeof(float4) ) );
	}

	void TimeIntegrationPcisphCUDAIhmsen2010Method(int currIndex, dataPointers& dptr, int& y)
	{
		// Note: here we use Ihmsen's boundary handling method, see Algorithm 2 from paper "Boundary handling and adaptive time-stepping for PCISPH"			
		TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep1<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex], 
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );	

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep1: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		//*
		TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep2<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex], dptr.particlesKVPair_d[1], 
			dptr.particlesKVPair_d[0] );

		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep2: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		//*/
		TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep3<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex], 
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: TimeIntegrationBlocksKernelPCISPHIhmsen2010MethodStep3: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Use the newly calculated pos_zindex array to bind as textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex], dptr.particleCountRounded*sizeof(float4) ) );
	}

	void TimeIntegrationStaticBoundariesPcisphCUDA(int currIndex, dataPointers& dptr, int& y)
	{
		// by default, we do not use Ihmsen's boundary handling method in PCISPH
		TimeIntegrationBlocksKernelStaticBoundariesPCISPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], dptr.d_vel[currIndex], 
			dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: TimeIntegrationBlocksKernelStaticBoundariesPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Use the newly calculated pos_zindex array to bind as textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex], dptr.particleCountRounded*sizeof(float4) ) );
	}

	void GetMaxPredictedDensityCUDA(dataPointers& dptr, float& max_predicted_density)
	{
		GetReductionMaxArray<float>(dptr.finalParticleCountRounded, dptr.param_num_threads_custom_reduction, 
			dptr.param_num_blocks_custom_reduction, (float*)dptr.d_predictedDensity, dptr.d_max_predicted_density_array);

		// TODO: Here we need to optimize this naive version
		GetReductionFinalMax<<<1,1>>>(dptr.d_max_predicted_density_array, dptr.param_num_blocks_custom_reduction, dptr.d_max_predicted_density_value);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: GetReductionFinalMax: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		cudaMemcpy(&max_predicted_density, dptr.d_max_predicted_density_value, sizeof(float), cudaMemcpyDeviceToHost);
	}

	void CorrectedDensityPressureComputationWallWeight(int currIndex, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays
		if (fc->physicsType == 'o')
		{
			CalculateCorrectedDensityPressureInBlocksKernelSPHWallWeight<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateCorrectedDensityPressureInBlocksKernelWCSPHWallWeight<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed density and pressure arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pressure, dptr.d_pressure, dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_density, dptr.d_density, dptr.particleCountRounded*sizeof(float) ) );

		dptr.filteredParticleCount = 0;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(filteredCount, &dptr.filteredParticleCount, sizeof(uint)) );
	}

	void ForceComputationVersatileCoupling(int currIndex, dataPointers& dptr, int& y)
	{
		// Previously the updated pressure & density arrays are stored in [currIndex] buffers & the corresponding textures are also binded to these arrays
		// But the pos & vel buffers are still stored in [1-currIndex] buffers
		if (fc->physicsType == 'o')
		{
			CalculateForcesPerParticleInBlocksKernelSPHVersatileCoupling<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesPerParticleInBlocksKernelSPHVersatileCoupling: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateForcesPerParticleInBlocksKernelWCSPHVersatileCoupling<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pos_zindex[currIndex], 
				dptr.d_vel[currIndex], dptr.d_smoothcolor_normal,
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesPerParticleInBlocksKernelWCSPHVersatileCoupling: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Now all the updated position & velocity arrays are stored in [currIndex] buffers
		// Bind the newly computed position arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex, dptr.d_pos_zindex[currIndex], dptr.particleCountRounded*sizeof(float4) ) );
	}

	void CalculateWeightedVolume(int currIndex, dataPointers& dptr, int& y)
	{
		CalculateWeightedVolumeInBlocksKernel<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_weighted_volume, dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateWeightedVolumeInBlocksKernel: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		CUDA_SAFE_CALL( cudaBindTexture(NULL, texture_weighted_volume, dptr.d_weighted_volume, dptr.particleCountRounded*sizeof(float) ) );
	}

	void CorrectedDensityPressureComputation(int currIndex, dataPointers& dptr, int& y)
	{
		// Here dptr.particlesKVPair_d[1] is an ancillary array of grid pairs(i.e. (grid_start_index, num_particles)) 
		// dptr.particlesKVPair_d[0] has the following layout: (currBlockIndex, currStartIndex, currNumParticles)
		// Now all the sorted attribute arrays are stored in [1-currIndex] buffers & the corresponding textures are also binded to these arrays
		if (fc->physicsType == 'o')
		{
			CalculateCorrectedDensitiesInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateCorrectedDensitiesInBlocksKernelWCSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_pressure,
				dptr.d_density, dptr.d_smoothcolor_normal, 
				dptr.particlesKVPair_d[1], dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateDensitiesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}

		// Bind the newly computed density and pressure arrays
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pressure, dptr.d_pressure, dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_density, dptr.d_density, dptr.particleCountRounded*sizeof(float) ) );

		dptr.filteredParticleCount = 0;
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(filteredCount, &dptr.filteredParticleCount, sizeof(uint)) );
	}


	void CalculateZindexRigidBody(int currIndex, dataPointers& dptr)
	{
		// Until Now all the updated attribute arrays except zindex buffers are stored in [currIndex] buffers
		FillZindicesKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_pos_zindex[1-currIndex], dptr.particlesKVPair_d[0]);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: FillZindicesKernel: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Finally we make sure all the updated attribute arrays are stored in [currIndex] buffers

	}

	void ForceComputationRigidFluidParticles(int currIndex, dataPointers& dptr, int& y)
	{
		// Previously the sorted pos & vel & type & parent_rb & relative_pos buffers are still stored in [1-currIndex] buffers
		if (fc->physicsType == 'o')
		{
			CalculateForcesBetweenRigidFluidParticlesInBlocksKernelSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_force,
																										 /*dptr.d_smoothcolor_normal,*/
																										  dptr.particlesKVPair_d[1], 
																										  dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesBetweenRigidFluidParticlesInBlocksKernelSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else if (fc->physicsType == 'w')
		{
			CalculateForcesBetweenRigidParticlesInBlocksKernelWCSPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>( dptr.d_force,
																									   /*dptr.d_smoothcolor_normal,*/
																									   dptr.particlesKVPair_d[1], 
																									   dptr.particlesKVPair_d[0] );

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: CalculateForcesBetweenRigidParticlesInBlocksKernelWCSPH: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
	}

	void UpdateRigidParticleIndicesArray(dataPointers& dptr)
	{
		UpdateRigidParticleIndicesKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_rigid_bodies);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: UpdateRigidParticleIndicesKernel: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();
	}

	void ZindexSortingRigidBodySPH(int currIndex, dataPointers& dptr)
	{
		// Sort the particle key-value pair(zindex, i) using radix-sort
		// dptr.particlesKVPair_d[0] is both the input and output array - data will be sorted
		// dptr.particlesKVPair_d[1] is just an additional array to allow ping pong computation
		RadixSort( (KeyValuePair*)dptr.particlesKVPair_d[0], (KeyValuePair*)dptr.particlesKVPair_d[1], dptr.particleCount, 32 );

		// Use the old attributes array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],								dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_relative_pos,					dptr.d_rigid_particle_relative_pos[currIndex],		dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[currIndex],								dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_parent_rb,						dptr.d_parent_rb[currIndex],						dptr.particleCountRounded*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_order_in_child_particles_array,	dptr.d_order_in_child_particles_array[currIndex],	dptr.particleCountRounded*sizeof(int) ) );

		CUDA_SAFE_CALL( cudaBindTextureToArray( texture_zindex_array, array_zindex ) );

		// Copy sorted particles values to the other ping pong array
		CopySortedParticleValuesKernelRigidBodySPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particleCountRounded, 
																						   dptr.d_pos_zindex[1-currIndex], 																						
																						   dptr.d_vel[1-currIndex], 
																					       dptr.d_rigid_particle_relative_pos[1-currIndex],
																						   dptr.d_type[1-currIndex], 
																						   dptr.d_parent_rb[1-currIndex], 
																						   dptr.d_order_in_child_particles_array[1-currIndex], 
																						   dptr.particlesKVPair_d[0] 
																						   );

		// Use the newly copied array to bind as texture
		// Now the sorted attributes arrays are stored in [1-currIndex] buffers
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_relative_pos,					dptr.d_rigid_particle_relative_pos[1-currIndex],	dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[1-currIndex],							dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_parent_rb,						dptr.d_parent_rb[1-currIndex],						dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_order_in_child_particles_array,	dptr.d_order_in_child_particles_array[1-currIndex],	dptr.particleCountRounded*sizeof(int) ) );	
	}

	void ZindexSortingStaticBoundariesPCISPH(int currIndex, int currType, dataPointers& dptr)
	{
		// Sort the particle key-value pair(zindex, i) using radix-sort
		// dptr.particlesKVPair_d[0] is both the input and output array - data will be sorted
		// dptr.particlesKVPair_d[1] is just an additional array to allow ping pong computation
		RadixSort( (KeyValuePair*)dptr.particlesKVPair_d[0], (KeyValuePair*)dptr.particlesKVPair_d[1], dptr.particleCount, 32 );

		// Use the old attributes array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],								dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[currType],								dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_corr_pressure,					dptr.d_correctionPressure,							dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_predicted_density,				dptr.d_predictedDensity,							dptr.particleCountRounded*sizeof(float) ) );

		CUDA_SAFE_CALL( cudaBindTextureToArray( texture_zindex_array, array_zindex ) );

		// Copy sorted particles values to the other ping pong array
		CopySortedParticleValuesKernelStaticBoundariesPCISPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particleCountRounded, 
																										dptr.d_pos_zindex[1-currIndex], 																						
																										dptr.d_vel[1-currIndex], 
																										dptr.d_type[1-currType], 
																										dptr.d_previous_correctionPressure,
																										dptr.d_previous_predicted_density,
																										dptr.particlesKVPair_d[0] 
																										);

		// Use the newly copied array to bind as texture
		// Now the sorted attributes arrays are stored in [1-currIndex] buffers
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[1-currType],							dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_previous_corr_pressure,			dptr.d_previous_correctionPressure,					dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_previous_predicted_density,		dptr.d_previous_predicted_density,					dptr.particleCountRounded*sizeof(float) ) );
	}

	void ZindexSortingRigidBodyPCISPH(int currIndex, dataPointers& dptr)
	{
		// Sort the particle key-value pair(zindex, i) using radix-sort
		// dptr.particlesKVPair_d[0] is both the input and output array - data will be sorted
		// dptr.particlesKVPair_d[1] is just an additional array to allow ping pong computation
		RadixSort( (KeyValuePair*)dptr.particlesKVPair_d[0], (KeyValuePair*)dptr.particlesKVPair_d[1], dptr.particleCount, 32 );

		// Use the old attributes array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],								dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_relative_pos,					dptr.d_rigid_particle_relative_pos[currIndex],		dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[currIndex],								dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_parent_rb,						dptr.d_parent_rb[currIndex],						dptr.particleCountRounded*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_order_in_child_particles_array,	dptr.d_order_in_child_particles_array[currIndex],	dptr.particleCountRounded*sizeof(int) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_corr_pressure,					dptr.d_correctionPressure,							dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_predicted_density,				dptr.d_predictedDensity,							dptr.particleCountRounded*sizeof(float) ) );

		CUDA_SAFE_CALL( cudaBindTextureToArray( texture_zindex_array, array_zindex ) );

		// Copy sorted particles values to the other ping pong array
		CopySortedParticleValuesKernelRigidBodyPCISPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.particleCountRounded, 
																								 dptr.d_pos_zindex[1-currIndex], 																						
																								 dptr.d_vel[1-currIndex], 
																								 dptr.d_rigid_particle_relative_pos[1-currIndex],
																								 dptr.d_type[1-currIndex], 
																								 dptr.d_parent_rb[1-currIndex], 
																								 dptr.d_order_in_child_particles_array[1-currIndex],
																								 dptr.d_previous_correctionPressure,
																								 dptr.d_previous_predicted_density,
																								 dptr.particlesKVPair_d[0] 
																								 );

		// Use the newly copied array to bind as texture
		// Now the sorted attributes arrays are stored in [1-currIndex] buffers
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_relative_pos,					dptr.d_rigid_particle_relative_pos[1-currIndex],	dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_type,							dptr.d_type[1-currIndex],							dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_parent_rb,						dptr.d_parent_rb[1-currIndex],						dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_order_in_child_particles_array,	dptr.d_order_in_child_particles_array[1-currIndex],	dptr.particleCountRounded*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_previous_corr_pressure,			dptr.d_previous_correctionPressure,					dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_previous_predicted_density,		dptr.d_previous_predicted_density,					dptr.particleCountRounded*sizeof(float) ) );
	}

	void ParticlePhysicsOnDeviceFluidRigidCouplingSPH(dataPointers& dptr)
	{
		// TODO: Problem exists in UpdateLiquidParticlePosVelSphGPU

		int currIndex = dptr.currIndex;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

		// Z-index and Sorting
		ZindexSortingRigidBodySPH(currIndex, dptr);

		// update rigid_particle_indices of rigid bodies
		// we have to do this because after sorting, the particle indices array in each rigid body
		// is invalid.
		UpdateRigidParticleIndicesArray(dptr);

		// Block Generation
		BlockGeneration(dptr, x, y);

		CalculateWeightedVolume(currIndex, dptr, y);

		// for each liquid particle, we compute its density & pressure using SPH/WCSPH/PCISPH
		if (fc->addWallWeightFunction)
		{
			CorrectedDensityPressureComputationWallWeight(currIndex, dptr, y);
		}
		else
		{
			CorrectedDensityPressureComputation(currIndex, dptr, y);
		}

		ForceComputationRigidFluidParticles(currIndex, dptr, y);

		// TODO: UpdateLiquidParticlePosVelSphGPU & RigidBodyIntegration in parallel
		UpdateLiquidParticlePosVelSphGPU(currIndex, dptr);

		// Three steps:
		//	(1)force & torque calculation
		//	(2)linear & angular momentum calculation
		//	(3)Rigid Body Integration
		RigidBodyIntegration(currIndex, dptr);

		// synchronize rigid particles with new updated rigid body positions
		SynRigidParticlesDevice(currIndex, dptr);

		//CalculateCorrectedForcePerParticle(dptr);

		UnbindTextureRigidFluidSPH();

		// Calculate Z-indices for particles
		CalculateZindexRigidBody(currIndex, dptr);

		// for sph, we don't need to switch the buffers for rendering 
		// Because now all the updated attribute arrays are stored in [currIndex] buffers
		PingPongSchemeRigidBody(dptr);
		
		// Synchronizing
		cudaDeviceSynchronize();
	}

	void ParticlePhysicsOnDeviceFluidRigidCouplingPCISPH(dataPointers& dptr)
	{

#ifdef SPH_PROFILING_VERBOSE
		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventRecord(startSimulation, 0);
#endif

		int currIndex = dptr.currIndex;
		// x : without redistribution
		// y : after redistribution
		int x = 0, y = 0;

		// Z-index and Sorting
		ZindexSortingRigidBodyPCISPH(currIndex, dptr);

		// update rigid_particle_indices of rigid bodies
		// we have to do this because after sorting, the particle indices array in each rigid body
		// is invalid.
		UpdateRigidParticleIndicesArray(dptr);

		// Block Generation
		BlockGeneration(dptr, x, y);

		CalculateWeightedVolume(currIndex, dptr, y);

		// External Force Computation
		//CalculateExternalForcesWithoutBoundaryForceFluidRigidCouplingPcisphCUDA(currIndex, dptr, y);
		//PredictionCorrectionStepPcisphCUDAVersatileCoupling(dptr, y);	

		CalculateExternalForcesFluidRigidCouplingPcisphCUDA(currIndex, dptr, y);

		PredictionCorrectionStepPcisphCUDA(dptr, y);

		//ComputeCorrectiveBoundaryFluidForcePcisphCUDA(dptr, y);

		UpdateLiquidParticlePosVelPCISPH(currIndex, dptr);

		// Three steps:
		//	(1)force & torque calculation
		//	(2)linear & angular momentum calculation
		//	(3)Rigid Body Integration
		RigidBodyIntegrationTwoWayCoupling(currIndex, dptr);

		// synchronize rigid particles with new updated rigid body positions
		SynRigidParticlesDevice(currIndex, dptr);

		UnbindTextureRigidFluidPCISPH();

		// Calculate Z-indices for particles
		CalculateZindexRigidBody(currIndex, dptr);

		// for sph, we don't need to switch the buffers for rendering 
		// Because now all the updated attribute arrays are stored in [currIndex] buffers
		PingPongSchemeRigidBody(dptr);

		// Synchronizing
		cudaDeviceSynchronize();

#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			float averageElapsed = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;

		}
#endif
	}

	void ParticlePhysicsOnDeviceFluidStaticBoundariesPCISPH(dataPointers& dptr)
	{	

#ifdef SPH_PROFILING_VERBOSE
		float elapsedTime;
		cudaEvent_t startSimulation, startSurfaceExtraction, stopSimulation, stopSurfaceExtraction;
		cudaEventCreate(&startSimulation);
		cudaEventCreate(&startSurfaceExtraction);
		cudaEventCreate(&stopSimulation);
		cudaEventCreate(&stopSurfaceExtraction);
		cudaEventRecord(startSimulation, 0);
#endif

		int currIndex = dptr.currIndex;
		int currType  = dptr.currType;
		// x : without redistribution
		// y : after redistribution
		int x = 0;
		int y = 0;

		// Z-index and Sorting
		ZindexSortingStaticBoundariesPCISPH(currIndex, currType, dptr);


		// Block Generation
		BlockGeneration(dptr, x, y);


		CalculateWeightedVolume(currIndex, dptr, y);


		// External Force Computation
		CalculateExternalForcesStaticBoundariesPcisphCUDA(currIndex, dptr, y);


		// prediction correction step
		PredictionCorrectionStepPcisphCUDA(dptr, y);


		// Time Integration
		TimeIntegrationStaticBoundariesPcisphCUDA(currIndex, dptr, y);


		// Extract Surface
		//ExtractSurface(dptr, x);


		// Calculate Z-indices for particles
		CalculateZindex(currIndex, dptr);

		UnbindTexturesPCISPH();

		PingPongScheme(dptr);

		// Synchronizing
		cudaDeviceSynchronize();

#ifdef SPH_PROFILING_VERBOSE
		cudaEventRecord(stopSimulation, 0);
		cudaEventSynchronize(stopSimulation);
		cudaEventElapsedTime(&elapsedTime, startSimulation, stopSimulation);
		simulationTimeCounter_ += elapsedTime / 1000.0f;
		cudaEventDestroy(startSimulation);
		cudaEventDestroy(stopSimulation);

		++frameCounter_;
		if (frameCounter_ % SPH_PROFILING_FREQ == 0)
		{
			float averageElapsed = simulationTimeCounter_ / frameCounter_;
			std::cout << "  physics simulation          average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
			averageElapsed = surfaceExtractionTimeCounter_ / frameCounter_;
			std::cout << "  surface particle extraction average: ";
			std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
			std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;

		}
#endif
	}

	void RigidBodyIntegration(int currIndex, dataPointers& dptr)
	{
		const int num_rigid_bodies = dptr.num_rigid_bodies;
		if (num_rigid_bodies > 0)
		{
			const int local_num_threads = 256;
			const int local_num_blocks = (num_rigid_bodies+ local_num_threads-1)/ local_num_threads;
			RigidBodyIntegrationKernel<<< local_num_blocks, local_num_threads>>>(currIndex,
																				 dptr.d_force,
																				 dptr.d_rigid_bodies
																				 );
		}

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr, "CUDA ERROR: RigidBodyIntegrationKernel: %s\n", cudaGetErrorString(error) );
		} 
		cudaDeviceSynchronize ();
	}

	void RigidBodyIntegrationTwoWayCoupling(int currIndex, dataPointers& dptr)
	{
		const int num_rigid_bodies = dptr.num_rigid_bodies;
		if (num_rigid_bodies > 0)
		{
			const int local_num_threads = 256;
			const int local_num_blocks = (num_rigid_bodies+ local_num_threads-1)/ local_num_threads;
			RigidBodyIntegrationTwoWayCouplingKernel<<< local_num_blocks, local_num_threads>>>(currIndex, dptr.d_rigid_bodies);
		}

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr, "CUDA ERROR: RigidBodyIntegrationTwoWayCouplingKernel: %s\n", cudaGetErrorString(error) );
		} 
		cudaDeviceSynchronize ();
	}

	void ComputeCorrectiveBoundaryFluidForcePcisphCUDA(dataPointers& dptr, int& y)
	{
		// we already bind pos & vel textures in the previous process, i.e. SynRigidParticlesDevice
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_corr_pressure,		 dptr.d_correctionPressure,  dptr.particleCountRounded*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL,	texture_predicted_density,   dptr.d_predictedDensity,	 dptr.particleCountRounded*sizeof(float4) ) );

		ComputeCorrectiveBoundaryFluidForceBlocksKernelPCISPH<<<y, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_dynamic_boundary_force, 
																								dptr.particlesKVPair_d[1], 
																								dptr.particlesKVPair_d[0] );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: ComputeCorrectiveBoundaryFluidForceBlocksKernelPCISPH: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();

		// Bind the newly updated correctionPressureForce textures
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_corr_pressure_force,		dptr.d_correctionPressureForce, dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_dynamic_boundary_force,  dptr.d_dynamic_boundary_force,  dptr.particleCountRounded*sizeof(float4) ) );
	}

	void SynRigidParticlesDevice(int currIndex, dataPointers& dptr)
	{
		const int num_rigid_bodies = dptr.num_rigid_bodies;
		if (num_rigid_bodies > 0)
		{
			// Previously the sorted pos & vel & type & parent_rb & relative_pos buffers are still stored in [1-currIndex] buffers
			SynRigidParticlesKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(currIndex,
																			   dptr.d_rigid_bodies,
																			   dptr.d_pos_zindex[1-currIndex],
																			   dptr.d_vel[1-currIndex]);

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				fprintf ( stderr,  "CUDA ERROR: SynRigidParticlesKernel: %s\n", cudaGetErrorString(error) );
			}    
			cudaDeviceSynchronize ();
		}
		else 
		{
			// if there aren't any rigid bodies, we still copy
			CUDA_SAFE_CALL( cudaMemcpy(dptr.d_pos_zindex[1-currIndex],  dptr.d_pos_zindex[currIndex],	dptr.particleCountRounded*sizeof(float4), cudaMemcpyDeviceToDevice ) );
			CUDA_SAFE_CALL( cudaMemcpy(dptr.d_vel[1-currIndex],			dptr.d_vel[currIndex],			dptr.particleCountRounded*sizeof(float4), cudaMemcpyDeviceToDevice ) );
		}

		// Use the newly updated pos & vel array to bind as texture
		//CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[1-currIndex],					dptr.particleCountRounded*sizeof(float4) ) );
		//CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[1-currIndex],						dptr.particleCountRounded*sizeof(float4) ) );	
	}

	void CalculateCorrectedForcePerParticle(dataPointers& dptr)
	{
		CalculateCorrectedForcePerParticleKernel<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(dptr.d_force);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr,  "CUDA ERROR: CalculateCorrectedForcePerParticleKernel: %s\n", cudaGetErrorString(error) );
		}    
		cudaDeviceSynchronize ();
	}

	void UpdateLiquidParticlePosVelSphGPU(int currIndex, dataPointers& dptr)
	{
		UpdateLiquidParticleKernelSPH<<<num_blocks, MAX_THREADS_PER_BLOCK_SPH>>>(currIndex,
																				 dptr.d_force,
																				 dptr.d_pos_zindex[currIndex],
																				 dptr.d_vel[currIndex]
																				 );

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf ( stderr, "CUDA ERROR: UpdateLiquidParticleKernelSPH: %s\n", cudaGetErrorString(error) );
		}
		cudaDeviceSynchronize ();

		// Use the newly updated pos & vel array to bind as texture
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_pos_zindex,						dptr.d_pos_zindex[currIndex],					dptr.particleCountRounded*sizeof(float4) ) );
		CUDA_SAFE_CALL( cudaBindTexture( NULL, texture_vel,								dptr.d_vel[currIndex],							dptr.particleCountRounded*sizeof(float4) ) );	
	}
}

extern "C"
	bool isPow2(unsigned int x)
{
	return ((x&(x-1))==0);
}

template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template<>
struct SharedMemory<float>
{
	__device__ inline operator       float *()
	{
		extern __shared__ float __smem_d[];
		return (float *)__smem_d;
	}

	__device__ inline operator const float *() const
	{
		extern __shared__ float __smem_d[];
		return (float *)__smem_d;
	}
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
	ReduceMax(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T myMax = 0;

	while (i < n)
	{
		myMax = fmaxf(myMax, g_idata[i]);

		if (nIsPow2 || i + blockSize < n)
			myMax = fmaxf(myMax, g_idata[i+blockSize]);

		i += gridSize;
	}

	sdata[tid] = myMax;
	__syncthreads();

	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  64]);
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		volatile T *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid + 32]);
		}

		if (blockSize >=  32)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid + 16]);
		}

		if (blockSize >=  16)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  8]);
		}

		if (blockSize >=   8)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  4]);
		}

		if (blockSize >=   4)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  2]);
		}

		if (blockSize >=   2)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  1]);
		}
	}

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void GetReductionMaxArray(uint size, uint threads, uint blocks, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			ReduceMax<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			ReduceMax<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			ReduceMax<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			ReduceMax<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			ReduceMax<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			ReduceMax<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			ReduceMax<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			ReduceMax<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			ReduceMax<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			ReduceMax<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
			ReduceMax<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			ReduceMax<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			ReduceMax<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			ReduceMax<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			ReduceMax<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			ReduceMax<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			ReduceMax<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			ReduceMax<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			ReduceMax<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			ReduceMax<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		}
	}

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ReduceMax: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}