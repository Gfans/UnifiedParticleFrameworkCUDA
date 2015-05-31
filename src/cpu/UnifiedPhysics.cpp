#include <omp.h>

#include "System/OpenGL.h"
#include "System/Profiling.h"
#include "System/Timer.h"

#include "zIndex.h"
#include "sort.h"
#include "UnifiedPhysics.h"
#include "Scene.h"
#include "render_particles.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm> 
#include <cfloat>
#include <limits>

#include <windows.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

void (*QuickSort)(std::vector< std::pair<unsigned int,unsigned int> >::iterator, std::vector< std::pair<unsigned int,unsigned int> >::iterator) 
	= QuickSortTemplate< std::pair<unsigned int,unsigned int>, 
	std::vector< std::pair<unsigned int,unsigned int> >::iterator, 
	std::less< std::pair<unsigned int,unsigned int> >, 
	PivotMedianOfThree< std::pair<unsigned int,unsigned int>, std::vector< std::pair<unsigned int,unsigned int> >::iterator, std::less< std::pair<unsigned int,unsigned int> > > 
	>;

static float diffuse[4] = {0.7f, 0.7f, 0.5f, 1.0f};
static float specular[4] = {1.0f, 1.0f, 1.0f, 1.0f};
static float shininess[1] = {8.0f};

ZIndex zIndex;

const float FluidMass = 0.05f;

const double kPi = 3.1415926535897932384626;

UnifiedPhysics::UnifiedPhysics(UnifiedConstants* _fc)
	:fc_(_fc),
	m_current_read_(0),
	m_current_write_(1)
#if defined(USE_VBO_CUDA) || defined(USE_VBO_CPU)
	 , renderer_(NULL)
#endif
	
#ifdef SPH_PROFILING
	,frame_counter_(0), 
	time_counter_(0.0f),
	time_counter_rendering_(0.0f),
	time_counter_total_(0.0f)
#endif
{
#ifdef USE_FFMPEG

	// start ffmpeg_ telling it to expect raw rgba 720p-60hz frames
	// -i - tells it to read frames from stdin
	const char* cmd = "ffmpeg_ -r 60 -f rawvideo -pix_fmt rgba -s 800x600 -i - "
		"-threads 0 -preset fast -y -crf 21 -vf vflip output.avi";

	// open pipe to ffmpeg_'s stdin in binary write mode
	ffmpeg_ = _popen(cmd, "wb");

	buffer_ = new int[800*600];

#endif

	iteration_counter_ = 0;
	elapsed_real_time_ = 0;

	CalculateKernelConstants();

	fc_->elapsedRealTime = 0.0;

	center_of_mass_static_boundary_box_.set(0.0, 0.0, 0.0);
	inertia_tensor_static_boundary_box.SetZero();

#ifdef SPH_DEMO_SCENE_2

	elapsed_real_time_ = 0.0;
	particle_count_ = 0;

#endif

#ifdef SPH_DEMO_SCENE_2
	fluidParticleSpacing = sizeFactor / 120.0;
	globalSupportRadius = fluidParticleSpacing * 2.0;

	deltaT = courantFactor * globalSupportRadius / relevantSpeed; 

	maxBoundaryForce = sizeFactor * 64 * fluidParticleSpacing * fluidParticleSpacing; //0.01;
	forceDistance = 0.25 * globalSupportRadius;

	initialVelocity.Set(2.0, 0.0, 0.0);
	//initialHeight = boxHeight*0.23;
	initialHeight = 0.23*4.0;

	pipeRadius = 0.1* sizeFactor;
	pipeHeight = box_length_ * 0.5 * 0.4;
	//pipePoint1.Set(-boxLength*0.5, initialHeight, boxWidth * 0.0);
	//pipePoint2.Set(-boxLength*0.5 + pipeHeight, initialHeight, boxWidth * 0.0);
	pipePoint1.Set(-0.9, 0.92, 0.0);
	pipePoint2.Set(-0.54, 0.92, 0.0);

	wallX = 0.7;
#endif

	my_kernel_ = new WeightingKernel(fc_);
	// precompute kernelValue
	kernel_self_ = my_kernel_->KernelM4(0.0f, fc_->globalSupportRadius);

	my_unified_io_ = new UnifiedIO();

	//fprintf(stderr, "To do:\t must convert to squared distance forms\n");
	//fprintf(stderr, "\t adjust kernel LUTs for squared distance\n");

	// before particles are created or read in from a file, we have to compute the densityErrorFactor for our incompressible method
	// always precompute correctionFactor -> makes switching between SPH/WCSPH/PCISPH easier and more efficient
	if (fc_->simulationMethod == 2)
	{
		// PCISPH
		ComputeDensityErrorFactor();
	}

	PrecomputeDistanceFunction();
	PrecomputeWallWeightFunction();

	// DEBUGGING
	//TestWallWeightFunction();

	// create scene
	const float spacing = fc_->particleSpacing;
	float jitter = 0.0f;
	int num_frozen_particles = 0;
	Scene scene;
	scene.build(this, fc_, spacing, jitter, num_frozen_particles);	

	// initiate parent_rigid_body_index & order_in_child_particles_array value for each particle
	const size_t num_rigid_bodies = rigid_bodies_.size();

	std::cout << std::endl << "***************************************************************************" << std::endl;
	for (int i = 0; i < num_rigid_bodies; ++i)
	{
		RigidBody* rb = rigid_bodies_[i];
		if (rb)
		{
			Matrix3x3::CalculateInverse(rb->inertia(), rb->inverted_inertia_local());
			const size_t num_rigid_particles = rb->rigid_particle_indices().size();
			for (int j = 0; j < num_rigid_particles; ++j)
			{
				const int rp_index = rb->rigid_particle_indices()[j];
				UnifiedParticle& p = particles_[rp_index];
				p.parent_rigid_body_index_ = i;	// we keep the rigid bodies in the same order as they are in rigidBodies vector
				p.order_in_child_particles_array_ = j;
				p.init_relative_pos_ = p.position_ - rb->rigidbody_pos();
			}		

		}
		std::cout << "The " << i+1 << "th rigid body's mass is : " << rb->mass() << std::endl;
	}
	std::cout << "***************************************************************************" << std::endl << std::endl;

	// calculate number of non frozen particles
	num_nonfrozen_particles_ = particles_.size() - num_frozen_particles;

#ifdef USE_VBO_CUDA

	DoCudaGLInteraction();

#endif

#if defined(HIDE_FROZEN_PARTICLES) && defined(USE_VBO_CPU)

	GlVBOInitWithoutFrozenParticles();

#else 

#ifdef USE_VBO_CPU

	GlVBOInit();

#endif	// end #ifdef USE_VBO_CPU

#endif // end #if defined(HIDE_FROZEN_PARTICLES) && defined(USE_VBO_CPU)



#if defined(USE_VBO_CUDA) || defined(USE_VBO_CPU)

	renderer_ = new ParticleRenderer;
	renderer_->setParticleRenderingSize(fc_->particleRenderingSize);
	renderer_->setColorBuffer(GetColorBuffer());

#endif

#ifdef USE_CUDA

	InitCudaVariables(particles_.size(), spacing);

#endif

	/* Code used for MLS
	vector<UnifiedParticle>::iterator endIter = particles.end();
	for (vector<UnifiedParticle>::iterator p = particles.begin(); p != endIter; ++p)
	{
	UnifiedParticle* currentParticle=&(*p);
	if(currentParticle->fluid!=FROZEN_PARTICLE)
	{
	CalculateInvertedMomentMatrix(currentParticle);
	}
	}
	//*/ 

	max_loops_sim_ = 0;
	avg_loops_sim_ = 0.0;
}

UnifiedPhysics::~UnifiedPhysics()
{
	particles_.clear();
	neighbor_indices_.clear();
	solid_neighs_.clear();
	solid_neigh_distances_.clear();

#ifdef USE_VBO_CUDA

	for (int i = 0; i < 2; ++i)
	{
		UnregisterGLBufferObject(cuda_posvbo_resource_[i]);
		glDeleteBuffers(1, (const GLuint *)&dptr_.m_posVbo[i]);
	}
	
	UnregisterGLBufferObject(cuda_colorvbo_resource_);
	glDeleteBuffers(1, &m_color_vbo_);	

#endif	

#ifdef USE_VBO_CPU

	glDeleteBuffers(1, (const GLuint *)&m_pos_vbo_cpu_);
	glDeleteBuffers(1, &m_color_vbo_);	

#endif

#if defined(USE_VBO_CUDA) || defined(USE_VBO_CPU)

	if (renderer_)
	{
		delete renderer_;
		renderer_ = NULL;
	}

	if (h_positions_)
	{
		delete [] h_positions_;
	}	

#endif

#ifdef HIDE_FROZEN_PARTICLES

	if (h_pos_without_frozen_)
	{
		delete [] h_pos_without_frozen_;
	}

#endif

	if (my_kernel_)
	{
		delete my_kernel_;
	}
	
	if (my_unified_io_)
	{
		delete my_unified_io_;
	}

	for(std::vector<RigidBody*>::iterator bodies=rigid_bodies_.begin(); bodies!=rigid_bodies_.end(); ++bodies)
	{
		delete (*bodies);
	}

#ifdef USE_FFMPEG

	if (buffer_)
	{
		delete buffer_;
		buffer_ = NULL;
	}

	_pclose(ffmpeg_);

#endif

	//FreeDeviceArrays(dptr_);

#ifdef USE_CUDA
	if (indices_g_)
	{
		delete[] indices_g_;
		indices_g_ = 0x0;
	}

	if (particle_info_for_rendering_.p_pos_zindex)
	{
		delete[] particle_info_for_rendering_.p_pos_zindex;
		particle_info_for_rendering_.p_pos_zindex = 0x0;
	}

	if (particle_info_for_rendering_.p_vel)
	{
		delete[] particle_info_for_rendering_.p_vel;
		particle_info_for_rendering_.p_vel = 0x0;
	}

	if (particle_info_for_rendering_.p_corr_pressure)
	{
		delete[] particle_info_for_rendering_.p_corr_pressure;
		particle_info_for_rendering_.p_corr_pressure = 0x0;
	}

	if (particle_info_for_rendering_.p_predicted_density)
	{
		delete[] particle_info_for_rendering_.p_predicted_density;
		particle_info_for_rendering_.p_predicted_density = 0x0;
	}

	if (particle_info_for_rendering_.p_type)
	{
		delete [] particle_info_for_rendering_.p_type;
		particle_info_for_rendering_.p_type = 0x0;
	}
	
	if (particle_info_for_rendering_.p_activeType)
	{
		delete [] particle_info_for_rendering_.p_activeType;
		particle_info_for_rendering_.p_activeType = 0x0;
	}

#endif	

#ifdef SPH_DEMO_SCENE_2
	delete[] ppf_pos_zindex_;
	delete[] ppf_vel_pressure_;
	delete[] ppf_zindex_;
#endif
}

void UnifiedPhysics::SortParticles()
{
	// There are some bugs with intro_sort in my case
	//Sort(particles.begin(), particles.end());	
	std::sort(particles_.begin(), particles_.end());
}

void UnifiedPhysics::UpdateRigidBodyParticleInformationOMP()
{
	int i;
	int chunk = 100;  
	// once we sort particles, we should also change the particles' parent rigid body information, i.e. parentRigidBody
	// iterate over all rigid bodies & clear buffers
	const unsigned int numRigidBodies = rigid_bodies_.size();
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait	
	for (int i = 0; i < numRigidBodies; ++i)
	{
		rigid_bodies_[i]->ClearBuffers();
	}

	// now we're ready for changing parent rigid body's information
	// iterate over all particles & make changes
	const unsigned int nofParticles = particles_.size();
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait	
	for (int j = 0; j < nofParticles; ++j)
	{
		RigidBody* parentRigid = particles_[j].parent_rigidbody_;
#pragma omp critical
		{
			if (parentRigid)
			{
				parentRigid->AddToParentRigidBody(j);
			}
		}
	}

	// iterate over all rigid bodies again and this time we update particle indices information & their new position
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait	
	for (int i = 0; i < numRigidBodies; ++i)
	{
		RigidBody* parentRigid = rigid_bodies_[i];
		if (parentRigid)
		{
			parentRigid->UpdateRigidBodyParticleIndices();
		}
	}
}

void UnifiedPhysics::CopyGPUParticles(UnifiedPhysics::ParticleInfoForRendering& particleInfo)
{
	uint count = CopyParticlesCUDA( particleInfo.p_pos_zindex, particleInfo.p_vel, particleInfo.p_corr_pressure, particleInfo.p_predicted_density, particleInfo.p_type, particleInfo.p_activeType, dptr_ );
}

#ifdef DUMP_PARTICLES_TO_FILE

void UnifiedPhysics::DumpGPUParamsToFile()
{
	ofstream wfile("output/particles.txt", ios::app);
	wfile.write((char*)&dptr_.particleCount, sizeof(uint32_t));
	wfile.write((char*)dptr_.cpuMem2, dptr_.particleCount*4*sizeof(float));
	wfile.close();

	uint dim = dptr_.dimX * dptr_.dimY * dptr_.dimZ;
	ofstream wfile1("output/blocks.txt", ios::app);
	wfile1.write((char*)&dptr_.filledBlocks, sizeof(uint32_t));
	int c = 0;
	for( uint32_t i = 0; i < dim; i++ )
	{
		if(dptr_.cpuMem1[2*i+1] > 0)
		{
			c++;
			wfile1.write((char*)&i, sizeof(uint32_t));
			// Starting index
			wfile1.write((char*)&dptr_.cpuMem1[2*i], sizeof(uint32_t));
			// Number
			wfile1.write((char*)&dptr_.cpuMem1[2*i+1], sizeof(uint32_t));
		}
	}

	//cout << dptr_.filledBlocks << "*" << endl;

	if( c != dptr_.filledBlocks )
	{
		int last = 100000;
		int c1 = 0;
		//cout << dptr_.filledBlocks << " " << c << " " << frame_counter_ << " " << dptr_.particleCount << endl;
		for( int i = 0; i < dptr_.particleCount; i++ )
		{
			int x = dptr_.cpuMem2[4*i] * fc_->scales.x / zIndex.block_size();
			int y = dptr_.cpuMem2[4*i+1] * fc_->scales.y / zIndex.block_size();
			int z = dptr_.cpuMem2[4*i+2] * fc_->scales.z / zIndex.block_size();
			int dest = (dptr_.dimX * dptr_.dimY) * z + (dptr_.dimX * y + x);
			if(dest != last)
			{
				//cout << dest << " " << x << " " << y << " " << z << " " << dptr_.cpuMem2[4*i+3] << " * ";
				if(dptr_.cpuMem1[2*dest] < i)
				{
					/*cout << dest << " " << dptr_.cpuMem2[4*i] * scales.x << " " << dptr_.cpuMem2[4*i+1] * scales.y << 
					" " << dptr_.cpuMem2[4*i+2] * scales.z << " " << dptr_.cpuMem2[4*i+3] << endl;
					cout << dest << " " << dptr_.cpuMem2[4*dest] * scales.x << " " << dptr_.cpuMem2[4*dest+1] * scales.y << 
					" " << dptr_.cpuMem2[4*dest+2] * scales.z << " " << dptr_.cpuMem2[4*dest+3] << endl;*/
				}
				c1++;
			}
			last = dest;
			//cout << dptr_.cpuMem2[4*i] << " " << dptr_.cpuMem2[4*i+1] << " " << dptr_.cpuMem2[4*i+2] << " " << dest << " * " << endl;
		}
		cout << endl;
		//cout << c1 << " * " << endl;
		/*for( uint32_t i = 0; i < dim; i++ )
		{
		if(dptr_.cpuMem1[2*i+1] > 0)
		cout << i << " " << dptr_.cpuMem1[2*i] << " " << dptr_.cpuMem1[2*i+1] << " * ";
		}*/
	}

	wfile1.close();
	//cout << endl;
}

#endif

void UnifiedPhysics::InitDptr(uint particleCount, float spacing)
{

	// Verify that fc is not NULL
	if (NULL == fc_)
	{
		fprintf(stderr, "fc is NULL!");
		exit(EXIT_FAILURE);
	}

	vmml::Vector3f dimensionsV				= vmml::Vector3f(GRID_RESOLUTION) / zIndex.block_size();

#ifdef SPH_DEMO_SCENE_2
	// All points need to be shifted by minimum of box
	cout << "Initializing Demo Scene 2 with max particles " << particleCount << endl;
	dptr_.demoScene						= 2;
	dptr_.particleCount					= 0;
	dptr_.finalParticleCount				= particleCount;
	dptr_.pipeRadius						= fc_->pipeRadius;
	dptr_.pipePoint1.x					= fc_->pipePoint1.x - virtualBoundingBox.getMin().x;
	dptr_.pipePoint1.y					= fc_->pipePoint1.y - virtualBoundingBox.getMin().y;
	dptr_.pipePoint1.z					= fc_->pipePoint1.z - virtualBoundingBox.getMin().z;
	dptr_.pipePoint2.x					= fc_->pipePoint2.x - virtualBoundingBox.getMin().x;
	dptr_.pipePoint2.y					= fc_->pipePoint2.y - virtualBoundingBox.getMin().y;
	dptr_.pipePoint2.z					= fc_->pipePoint2.z - virtualBoundingBox.getMin().z;
	dptr_.wallX							= fc_->wallX;
	dptr_.fluidViscosityConstant_tube	= fc_->fluidViscConst_tube;
#else
	std::cout << "Initializing Demo Scene 1 " << std::endl;
	dptr_.demoScene						= 1;
	dptr_.particleCount					= particleCount;
	dptr_.finalParticleCount				= particleCount;
	int temp_cout						= particleCount; 
	if( particleCount % MAX_THREADS_PER_BLOCK_SPH )
		temp_cout += ( MAX_THREADS_PER_BLOCK_SPH - temp_cout % MAX_THREADS_PER_BLOCK_SPH );
	dptr_.particleCountRounded			= temp_cout;
#endif
	dptr_.maxLength						= std::max((float)dimensionsV.x*dimensionsV.y*dimensionsV.z, (float)particleCount);
	dptr_.filteredParticleCount			= 0;
	dptr_.globalSupportRadius			= fc_->globalSupportRadius;
	dptr_.distToCenterMassCutoff			= fc_->distToCenterMassCutoff;
	dptr_.gradientVelCutoff				= fc_->gradientVelCutoff;
	dptr_.particleRadius					= fc_->particleRadius;
	dptr_.rb_spring						= fc_->springCoefficient;
	dptr_.rb_spring_boundary				= fc_->springCoefficientBoundary;
	dptr_.rb_damping						= fc_->dampingCoefficient;
	dptr_.surface_tension_gamma		    = fc_->surface_tension_gamma;
	dptr_.surface_adhesion_beta			= fc_->surface_adhesion_beta;
	dptr_.rb_terminalSpeed				= fc_->terminalSpeed;
	dptr_.particleRenderingSize  		= fc_->particleRenderingSize;
	dptr_.grid_resolution				= GRID_RESOLUTION;
	dptr_.block_size						= zIndex.block_size();
	dptr_.lutSize						= WeightingKernel::lut_size();
	dptr_.kernelSelf						= kernel_self_;
	dptr_.initialMass					= fc_->initialMass;
	dptr_.fluidRestDensity				= fc_->fluidRestDensity;
	dptr_.gamma							= fc_->gamma;
	dptr_.fluidGasConstant				= fc_->fluidGasConst;
	dptr_.fluidGasConstantWCSPH			= fc_->fluidGasConstantWCSPH;
	dptr_.addBoundaryForce				= fc_->addBoundaryForce;
	dptr_.forceDistance					= fc_->forceDistance;
	dptr_.maxBoundaryForce				= fc_->maxBoundaryForce;
	dptr_.minCollisionBox.x				= fc_->collisionBox.getMin().x;
	dptr_.minCollisionBox.y				= fc_->collisionBox.getMin().y;
	dptr_.minCollisionBox.z				= fc_->collisionBox.getMin().z;
	dptr_.maxCollisionBox.x				= fc_->collisionBox.getMax().x;
	dptr_.maxCollisionBox.y				= fc_->collisionBox.getMax().y;
	dptr_.maxCollisionBox.z				= fc_->collisionBox.getMax().z;	
	dptr_.minBoundingBox.x				= fc_->virtualBoundingBox.getMin().x;
	dptr_.minBoundingBox.y 				= fc_->virtualBoundingBox.getMin().y;
	dptr_.minBoundingBox.z 				= fc_->virtualBoundingBox.getMin().z;
	dptr_.maxBoundingBox.x  				= fc_->virtualBoundingBox.getMax().x;
	dptr_.maxBoundingBox.y  				= fc_->virtualBoundingBox.getMax().y;
	dptr_.maxBoundingBox.z  				= fc_->virtualBoundingBox.getMax().z;	
	dptr_.minContainerBox.x  			= fc_->realBoxContainer.getMin().x; 
	dptr_.minContainerBox.y  			= fc_->realBoxContainer.getMin().y;
	dptr_.minContainerBox.z  			= fc_->realBoxContainer.getMin().z;
	dptr_.maxContainerBox.x  			= fc_->realBoxContainer.getMax().x;
	dptr_.maxContainerBox.y  			= fc_->realBoxContainer.getMax().y;
	dptr_.maxContainerBox.z  			= fc_->realBoxContainer.getMax().z;
	dptr_.boxLength  					= fc_->boxLength;
	dptr_.boxHeigth  					= fc_->boxHeight;
	dptr_.boxWidth  						= fc_->boxWidth;
	dptr_.zindexStartingVec.x 			= fc_->zindexStartingVec.x;
	dptr_.zindexStartingVec.y 			= fc_->zindexStartingVec.y;
	dptr_.zindexStartingVec.z 			= fc_->zindexStartingVec.z;
	dptr_.dimX 							= dimensionsV.x;
	dptr_.dimY 							= dimensionsV.y;
	dptr_.dimZ 							= dimensionsV.z;
	dptr_.deltaT 						= fc_->deltaT;
	dptr_.deltaTWCSPH					= fc_->deltaT_wcsph;
	dptr_.scales.x						= fc_->scales.x;
	dptr_.scales.y						= fc_->scales.y;
	dptr_.scales.z						= fc_->scales.z;
	dptr_.gravityConstant				= fc_->gravityConst;
	dptr_.fluidViscosityConstant			= fc_->fluidViscConst;
	dptr_.spacing						= spacing;
	dptr_.currIndex						= 0;
	dptr_.currType						= 0;
	dptr_.currActiveType					= 0;	
	//dptr_.centerOfMassThreshold			= sr::Settings::instance()->getCenterOfMassThreshold();
	//dptr_.nNeighborsThreshold			= sr::Settings::instance()->getNNeighborsThreshold();
	dptr_.num_rigid_bodies				= rigid_bodies_.size();
	// PCISPH
	dptr_.param_density_error_factor		= fc_->densityErrorFactor;
	dptr_.param_min_loops				= MIN_PCISPH_LOOPS;
	dptr_.param_max_loops				= MAX_PCISPH_LOOPS;
	dptr_.param_max_density_error_allowed = 1.0;	// 1%

	//neighbor_indices					= new uint[ 2*particleCount ];

	indices_g_							= new uint32_t[3*dptr_.maxLength];
	uint sizeofFluidParticles			= sizeof(UnifiedParticle);
	
#ifdef USE_VBO_CUDA
	particle_info_for_rendering_.p_pos_zindex			= NULL;
	particle_info_for_rendering_.p_vel					= NULL;
	particle_info_for_rendering_.p_corr_pressure		= NULL;
	particle_info_for_rendering_.p_predicted_density	= NULL;
	particle_info_for_rendering_.p_type					= NULL;
	particle_info_for_rendering_.p_activeType			= NULL;
#else
	particle_info_for_rendering_.p_pos_zindex			= new float[sizeof(float4)*dptr_.particleCountRounded];
	particle_info_for_rendering_.p_vel					= new float[sizeof(float4)*dptr_.particleCountRounded];
	particle_info_for_rendering_.p_corr_pressure		= new float[dptr_.particleCountRounded];
	particle_info_for_rendering_.p_predicted_density	= new float[dptr_.particleCountRounded];
	particle_info_for_rendering_.p_type					= new int[dptr_.particleCountRounded];
	particle_info_for_rendering_.p_activeType			= new int[dptr_.particleCountRounded];
#endif

#ifdef DUMP_PARTICLES_TO_FILE
	dptr_.cpuMem1						= indices_g_;
	dptr_.cpuMem2						= particle_info_for_rendering_.p_pos_zindex;
#endif

}

void UnifiedPhysics::InitCudaVariables(size_t particleCount, float spacing)
{
	InitDptr(particleCount, spacing);

	// Please do not change the sequence of these functions
	MallocDeviceArraysParticles( dptr_ );
	MallocDeviceArraysRigidBody( dptr_, rigid_bodies_ );
	SetDeviceConstants( dptr_ );
	CreateZIndexTexture( zIndex.z_table() );
	CreateLutKernelM4Texture( my_kernel_->lut_kernel_m4(), WeightingKernel::lut_size() );
	CreateLutKernelPressureGradientTexture( my_kernel_->lut_kernel_pressure_grad(), WeightingKernel::lut_size() );
	CreateLutKernelViscosityLapTexture(my_kernel_->lut_kernel_viscosity_lap(), WeightingKernel::lut_size() );
	CreateLutSplineSurfaceTensionTexture(my_kernel_->lut_spline_surface_tension(), WeightingKernel::lut_size());
	CreateLutSplineSurfaceAdhesionTexture(my_kernel_->lut_spline_surface_adhesion(), WeightingKernel::lut_size());


#ifdef SPH_DEMO_SCENE_2

	//CopyAllParticleLayersHostToDevice( dptr_, ppf_pos_zindex_, ppf_vel_pressure_, ppf_zindex_, ppf_particle_count_ );

#else

	CopyZIndicesKVPairHostToDevice( dptr_, particles_ );
	CopyParticleDataHostToDevice( dptr_, particles_);
	CopyRigidBodyDataHostToDevice(rigid_bodies_, dptr_);

#endif

}

#ifdef SPH_DEMO_SCENE_2	
// One time process to malloc memory and create particle layer
// These same set of particles are added to the SPH pipeline
// in every frame
void UnifiedPhysics::CreateParticleLayer()
{
	float currSpacing = fc_->particleSpacing;
	float transX = -(fc_->boxLength - fc_->fluidLength)*0.5;
	float transY = fc_->initialHeight - fc_->fluidHeight * 0.5;
	float transZ = 0.0;

	float jitter = 0.005;

	int numParticlesX = (int)(fc_->fluidLength / currSpacing + 0.5);
	int numParticlesY = (int)(fc_->fluidHeight / currSpacing + 0.5);
	int numParticlesZ = (int)(fc_->fluidWidth  / currSpacing + 0.5);
	float myBL = numParticlesX * currSpacing;	
	float myBW = numParticlesZ * currSpacing;

	//cout << "particle layer " << endl;
	//cout << numParticlesX << " " << numParticlesY << " " << numParticlesZ << " " << zIndex.block_size() << endl;

	float tmpX, tmpY, tmpZ;
	if(numParticlesX % 2 == 0)
		tmpX = 0.0;
	else
		tmpX = 0.5;
	if(numParticlesZ % 2 == 0)
		tmpZ = 0.0;
	else
		tmpZ = 0.5;

	ppf_particle_count_ = numParticlesX * numParticlesY * numParticlesZ;

	// Additional space for dummy particles for CUDA kernels
	ppf_pos_zindex_ = new float[4*(ppf_particle_count_+128)];
	ppf_vel_pressure_ = new float[4*(ppf_particle_count_+128)];
	ppf_zindex_ = new unsigned int[2*(ppf_particle_count_+128)];
	vmml::Vector3f deltaT = fc_->virtualBoundingBox.getMin().x - fc_->virtualBoundingBox.getMax().x;

	vmml::Vector3f delta2 = fc_->virtualBoundingBox.getMax() - fc_->virtualBoundingBox.getMin();

	// check if inside pipe
	float pipeRadius = fc_->pipeRadius;
	Vector3<float> x1 = fc_->pipePoint1;
	x1.x -= fc_->virtualBoundingBox.getMin().x;
	x1.y -= fc_->virtualBoundingBox.getMin().y;
	x1.z -= fc_->virtualBoundingBox.getMin().z;
	Vector3<float> x2 = fc_->pipePoint2;
	x2.x -= fc_->virtualBoundingBox.getMin().x;
	x2.y -= fc_->virtualBoundingBox.getMin().y;
	x2.z -= fc_->virtualBoundingBox.getMin().z;

	int totalP = 0;
	for(int iy = 0; iy < numParticlesY; iy++)
	{
		float y = 0.0 + (iy + 0.5) * currSpacing;
		for(int ix = 0; ix < numParticlesX; ix++)
		{
			float x = -myBL/2 + (ix + tmpX) * currSpacing;
			for(int iz = 0; iz < numParticlesZ; iz++)
			{	
				float z = -myBW/2 + (iz + tmpZ) * currSpacing;		

				float px = x + transX + ((drand48() - 0.5) * jitter);
				float py = y + transY + ((drand48() - 0.5) * jitter);
				float pz = z + transZ + ((drand48() - 0.5) * jitter);


				px -= fc_->virtualBoundingBox.getMin().x;
				//px += 0.1;
				py -= fc_->virtualBoundingBox.getMin().y;
				pz -= fc_->virtualBoundingBox.getMin().z;
				//pz += 0.3;
				Vector3<float> position(px, py, pz);

				// compute point-line distance in 3D

				float val1 = ((x2 - x1).cross(x1 - position)).length();
				Vector3<float> axis = x2 - x1;
				float val2 = axis.length();
				float d = val1 / val2;

				// check if inside or outside
				// if outside do not create particle
				if(d > fc_->pipeRadius) 
					continue;

				/*UnifiedParticle p;
				p.position.x = px;
				p.position.y = py;
				p.position.z = pz;
				p.index = zIndex.CalcIndex(scales * Vector3<float>(
				p.position.x,
				p.position.y,
				p.position.z) );

				p.velocity.x = fc->initialVelocity[0];
				p.velocity.y = fc->initialVelocity[1];
				p.velocity.z = fc->initialVelocity[2];
				particles.push_back(p);*/

				//vmml::Vector3f grid(px, py, pz);
				//grid = grid * scales / zIndex.block_size();
				//cout << grid << endl;

				ppf_pos_zindex_[4*totalP] = px;
				ppf_pos_zindex_[4*totalP+1] = py;
				ppf_pos_zindex_[4*totalP+2] = pz;
				ppf_pos_zindex_[4*totalP+3] = zIndex.CalcIndex(fc_->scales * Vector3<float>(
					ppf_pos_zindex_[4*totalP],
					ppf_pos_zindex_[4*totalP+1],
					ppf_pos_zindex_[4*totalP+2]) );
				ppf_zindex_[2*totalP] = zIndex.CalcIndex(fc_->scales * Vector3<float>(
					ppf_pos_zindex_[4*totalP],
					ppf_pos_zindex_[4*totalP+1],
					ppf_pos_zindex_[4*totalP+2]) );
				ppf_zindex_[2*totalP+1] = totalP; 
				//cout << ppf_pos_zindex_[4*totalP] << " " << ppf_pos_zindex_[4*totalP+1] << " "
				//  << ppf_pos_zindex_[4*totalP+2] << " " << ppf_pos_zindex_[4*totalP+3] << endl;


				ppf_vel_pressure_[4*totalP] = fc_->initialVelocity[0];
				ppf_vel_pressure_[4*totalP+1] = fc_->initialVelocity[1];
				ppf_vel_pressure_[4*totalP+2] = fc_->initialVelocity[2];
				ppf_vel_pressure_[4*totalP+3] = 0.0;

				totalP++;
				//cout << px << " " << py << " " << pz << endl;

			}
		}
	}

	ppf_particle_count_ = totalP;

	//*
	cout << "Fluid length : " << fc_->fluidLength << " " 
		<< "Fluid Height : " << fc_->fluidHeight << " " 
		<< " Fluid Width : " << fc_->fluidWidth << endl;
	cout << ppf_particle_count_ << " " << fc_->particleSpacing << endl;
	cout << "Pipe points : " << x1 << " " << x2 << endl;
	cout << "Time step : " << fc_->deltaT << endl;
	cout << "Particle spacing : " << fc_->particleSpacing << endl;
	cout << "Global support radius : " << fc_->globalSupportRadius << endl;
	cout << "Particle mass : " << fc_->initialMass << endl;
	cout << "Gas constant : " << fc_->fluidGasConst << endl;
	cout << "Visc constant : " << fc_->fluidViscConst << endl;
	cout << "Force Distance : " << fc_->forceDistance << endl;
	//*/
}
#endif

//--------------------------------------------------------------------
void UnifiedPhysics::SolidifyRange(const int startIndex, const int endIndex)
{
	UpdateNeighborsRangeParallel(startIndex, endIndex);

	for(int particleIndex = startIndex; particleIndex < endIndex; ++particleIndex)
	{		
		FreezeParticle(particleIndex);
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::SolidifyParticle(const int i)
{	
	FreezeParticle(i);
}

//--------------------------------------------------------------------
void UnifiedPhysics::Add(UnifiedParticle* newParticle)
{
	// (Eq. 6.5) from "Simulation of Fluid-Solid Interaction"
	// use kernelSelf to replace KernelDensity(0)
	newParticle->solid_volume_ = 1 / kernel_self_; 
	particles_.push_back(*newParticle);
}

//--------------------------------------------------------------------
void UnifiedPhysics::AddToRigidBody(RigidBody* rigid, const int pIndex)
{	
	UnifiedParticle& p = particles_[pIndex];
	const float partMass = p.particle_mass_;
	float newMass = rigid->mass() + partMass;
	vmml::Vector3f newCenterOfMass = (rigid->rigidbody_pos()*rigid->mass()+p.position_*partMass)/newMass;	// (2-18) from David Baraff's "Physically Based Modeling Rigid Body Simulation"
	vmml::Vector3f distCenters = rigid->rigidbody_pos() - newCenterOfMass; // distance between old and new center of mass
	vmml::Vector3f distCenterPart = p.position_ - newCenterOfMass; // distance between new p and new center of mass
	//p.init_relative_pos = distCenterPart;	// we update initial relative position after adding new particles in here TODO: is this correct?
	// measuring inertia in a frame of reference of the new CoM:
	// inertia.elements[0][0] += mass_ * (distCenters[1]*distCenters[1] + distCenters[2]*distCenters[2])
	// inertia.elements[0][1] = inertia.elements[1][0] -= mass_ * distCenters[0]*distCenters[1];
	// and so on.
	// adding the new p to the modified inertia:
	// inertia.elements[0][0] += partMass * (distCenterPart[1]*distCenterPart[1] + distCenterPart[2]*distCenterPart[2])
	// inertia.elements[0][1] = inertia.elements[1][0] -= partMass * distCenterPart[0]*distCenterPart[1];
	// and so on.
	// doing everything at the same time:
	float centersSq0 = distCenters[0]*distCenters[0];
	float centersSq1 = distCenters[1]*distCenters[1];
	float centersSq2 = distCenters[2]*distCenters[2];
	float partSq0 = distCenterPart[0]*distCenterPart[0];
	float partSq1 = distCenterPart[1]*distCenterPart[1];
	float partSq2 = distCenterPart[2]*distCenterPart[2];

	Matrix3x3 tmp_inertia(rigid->inertia());
	tmp_inertia.elements[0][0] += rigid->mass() * (centersSq1 + centersSq2) + partMass * (partSq1 + partSq2);
	tmp_inertia.elements[1][1] += rigid->mass() * (centersSq0 + centersSq2) + partMass * (partSq0 + partSq2);
	tmp_inertia.elements[2][2] += rigid->mass() * (centersSq1 + centersSq0) + partMass * (partSq1 + partSq0);
	tmp_inertia.elements[0][1] -= rigid->mass() * distCenters[0]*distCenters[1] + partMass * distCenterPart[0]*distCenterPart[1];
	tmp_inertia.elements[0][2] -= rigid->mass() * distCenters[0]*distCenters[2] + partMass * distCenterPart[0]*distCenterPart[2];
	tmp_inertia.elements[1][2] -= rigid->mass() * distCenters[1]*distCenters[2] + partMass * distCenterPart[1]*distCenterPart[2];
	tmp_inertia.elements[1][0] = tmp_inertia.elements[0][1];
	tmp_inertia.elements[2][0] = tmp_inertia.elements[0][2];
	tmp_inertia.elements[2][1] = tmp_inertia.elements[1][2];
	rigid->set_inertia(tmp_inertia);

	// update initial local inertia tensor
	//Matrix3x3::CalculateInverse(rigid->inertia, rigid->invertedInertiaLocal);

	// adapt angular momentum:
	// angular momentum in the new frame of reference is
	// angularMomentum ("local term") + angular momentum due to rotation about the new frame ("remote term")
	// remote term = distCenters X linear momentum
	rigid->set_angular_momentum(rigid->angular_momentum() + distCenters.cross(rigid->velocity()*rigid->mass()));
	// add the angular momentum of p:
	rigid->angular_momentum() += distCenterPart.cross(p.velocity_*partMass);

	// adapt linear velocity
	const vmml::Vector3f vel = (rigid->velocity()*rigid->mass() + p.velocity_ * partMass) / newMass;
	rigid->set_velocity(vel); 

	// set pointers
	rigid->rigid_particle_indices().push_back(pIndex);
	p.parent_rigidbody_ = rigid;

	rigid->set_rigidbody_pos(newCenterOfMass);
	rigid->set_mass(newMass);
}

//--------------------------------------------------------------------
void UnifiedPhysics::AddRigidBodyParticles(RigidBody* currentRigid, RigidBody* part)
{	
	const float partMass = part->mass();
	float newMass = currentRigid->mass() + partMass;
	vmml::Vector3f newCenterOfMass = (currentRigid->rigidbody_pos()*currentRigid->mass()+part->rigidbody_pos()*partMass)/newMass;
	vmml::Vector3f distCenters = currentRigid->rigidbody_pos() - newCenterOfMass; // distance between old and new center of mass
	vmml::Vector3f distCenterPart = part->rigidbody_pos() - newCenterOfMass; // distance between new part and new center of mass
	//TODO: update its particles's new init_relative_pos
	// measuring inertia in a frame of reference of the new CoM:
	// inertia.elements[0][0] += mass_ * (distCenters[1]*distCenters[1] + distCenters[2]*distCenters[2])
	// inertia.elements[0][1] = inertia.elements[1][0] -= mass_ * distCenters[0]*distCenters[1];
	// and so on.
	// measuring the inertia of part in a frame of reference of the new CoM:
	// part->inertia.elements[0][0] += partMass * (distCenterPart[1]*distCenterPart[1] + distCenterPart[2]*distCenterPart[2])
	// part->inertia.elements[0][1] = part->inertia.elements[1][0] -= partMass * distCenterPart[0]*distCenterPart[1];
	// and so on.
	// final inertia: inertia = inertia + part->inertia
	// doing everything at the same time:
	float centersSq0 = distCenters[0]*distCenters[0];
	float centersSq1 = distCenters[1]*distCenters[1];
	float centersSq2 = distCenters[2]*distCenters[2];
	float partSq0 = distCenterPart[0]*distCenterPart[0];
	float partSq1 = distCenterPart[1]*distCenterPart[1];
	float partSq2 = distCenterPart[2]*distCenterPart[2];

	Matrix3x3 tmp_inertia(currentRigid->inertia());
	tmp_inertia.elements[0][0] += currentRigid->mass() * (centersSq1 + centersSq2) + part->inertia().elements[0][0] + partMass * (partSq1 + partSq2);
	tmp_inertia.elements[1][1] += currentRigid->mass() * (centersSq0 + centersSq2) + part->inertia().elements[1][1] + partMass * (partSq0 + partSq2);
	tmp_inertia.elements[2][2] += currentRigid->mass() * (centersSq1 + centersSq0) + part->inertia().elements[2][2] + partMass * (partSq1 + partSq0);
	tmp_inertia.elements[0][1] += -currentRigid->mass() * distCenters[0]*distCenters[1] + part->inertia().elements[0][1] - partMass * distCenterPart[0]*distCenterPart[1];
	tmp_inertia.elements[0][2] += -currentRigid->mass() * distCenters[0]*distCenters[2] + part->inertia().elements[0][2] - partMass * distCenterPart[0]*distCenterPart[2];
	tmp_inertia.elements[1][2] += -currentRigid->mass() * distCenters[1]*distCenters[2] + part->inertia().elements[1][2] - partMass * distCenterPart[1]*distCenterPart[2];
	tmp_inertia.elements[1][0] = tmp_inertia.elements[0][1];
	tmp_inertia.elements[2][0] = tmp_inertia.elements[0][2];
	tmp_inertia.elements[2][1] = tmp_inertia.elements[1][2];
	currentRigid->set_inertia(tmp_inertia);

	// adapt angular momentum:
	// angular momentum in the new frame of reference is
	// angularMomentum ("local term") + angular momentum due to rotation about the new frame ("remote term")
	// remote term = distCenters X linear momentum
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() + distCenters.cross(currentRigid->velocity()*currentRigid->mass()));
	// add the angular momentum of part:
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() + part->angular_momentum() + distCenterPart.cross(part->velocity()*partMass));

	// adapt linear velocity
	const vmml::Vector3f vel = (currentRigid->velocity()*currentRigid->mass() + part->velocity() * partMass) / newMass;
	currentRigid->set_velocity(vel); 

	// add particles & set pointers
	std::vector<int> &temp = currentRigid->rigid_particle_indices();
	int firstNewElement = temp.size();
	temp.insert(temp.end(), part->rigid_particle_indices().begin(), part->rigid_particle_indices().end());
	for(int i=firstNewElement; i<temp.size(); ++i)
	{
		particles_[temp[i]].parent_rigidbody_ = currentRigid;
	}
	currentRigid->set_rigidbody_pos(newCenterOfMass);
	currentRigid->set_mass(newMass);
}


//--------------------------------------------------------------------
void UnifiedPhysics::RemoveFromRigidBody(RigidBody* currentRigid, UnifiedParticle* part)
{	
	float partMass = part->particle_mass_;
	float newMass = currentRigid->mass() - partMass;
	vmml::Vector3f newCenterOfMass = (currentRigid->rigidbody_pos()*currentRigid->mass()-part->position_*partMass)/newMass;
	vmml::Vector3f distCenters = newCenterOfMass - currentRigid->rigidbody_pos(); // distance between new and old center of mass
	vmml::Vector3f distCenterPart = vmml::Vector3f(part->position_) - currentRigid->rigidbody_pos(); // distance between part and old center of mass
	// removing part from inertia:
	// inertia.elements[0][0] -= partMass * (distCenterPart[1]*distCenterPart[1] + distCenterPart[2]*distCenterPart[2])
	// inertia.elements[0][1] = inertia.elements[1][0] += partMass * distCenterPart[0]*distCenterPart[1];
	// and so on.
	// measuring inertia in a frame of reference of the new CoM:
	// inertia.elements[0][0] -= newMass * (distCenters[1]*distCenters[1] + distCenters[2]*distCenters[2])
	// inertia.elements[0][1] = inertia.elements[1][0] += newMass * distCenters[0]*distCenters[1];
	// and so on.
	// doing everything at the same time:
	float centersSq0 = distCenters[0]*distCenters[0];
	float centersSq1 = distCenters[1]*distCenters[1];
	float centersSq2 = distCenters[2]*distCenters[2];
	float partSq0 = distCenterPart[0]*distCenterPart[0];
	float partSq1 = distCenterPart[1]*distCenterPart[1];
	float partSq2 = distCenterPart[2]*distCenterPart[2];

	Matrix3x3 tmp_inertia(currentRigid->inertia());
	tmp_inertia.elements[0][0] -= newMass * (centersSq1 + centersSq2) + partMass * (partSq1 + partSq2);
	tmp_inertia.elements[1][1] -= newMass * (centersSq0 + centersSq2) + partMass * (partSq0 + partSq2);
	tmp_inertia.elements[2][2] -= newMass * (centersSq1 + centersSq0) + partMass * (partSq1 + partSq0);
	tmp_inertia.elements[0][1] += newMass * distCenters[0]*distCenters[1] + partMass * distCenterPart[0]*distCenterPart[1];
	tmp_inertia.elements[0][2] += newMass * distCenters[0]*distCenters[2] + partMass * distCenterPart[0]*distCenterPart[2];
	tmp_inertia.elements[1][2] += newMass * distCenters[1]*distCenters[2] + partMass * distCenterPart[1]*distCenterPart[2];
	tmp_inertia.elements[1][0] = tmp_inertia.elements[0][1];
	tmp_inertia.elements[2][0] = tmp_inertia.elements[0][2];
	tmp_inertia.elements[2][1] = tmp_inertia.elements[1][2];
	currentRigid->set_inertia(tmp_inertia);

	// adapt linear velocity
	const vmml::Vector3f vel = (currentRigid->velocity()*currentRigid->mass() - part->velocity_ * partMass) / newMass;
	currentRigid->set_velocity(vel); 

	// adapt angular momentum:
	// remove the angular momentum of part:
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() - distCenterPart.cross(part->velocity_*partMass));
	// move the frame of reference
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() - distCenters.cross(currentRigid->velocity()*newMass));

	currentRigid->set_rigidbody_pos(newCenterOfMass);
	currentRigid->set_mass(newMass);
}

//--------------------------------------------------------------------
void UnifiedPhysics::RemoveRigidBodyParticles(RigidBody* currentRigid, RigidBody* part)
{	
	float partMass = part->mass();
	float newMass = currentRigid->mass() - partMass;
	vmml::Vector3f newCenterOfMass = (currentRigid->rigidbody_pos()*currentRigid->mass()-part->rigidbody_pos()*partMass)/newMass;
	vmml::Vector3f distCenters = newCenterOfMass - currentRigid->rigidbody_pos(); // distance between new and old center of mass
	vmml::Vector3f distCenterPart = part->rigidbody_pos() - currentRigid->rigidbody_pos(); // distance between part and old center of mass
	// removing part from inertia:
	// inertia.elements[0][0] -= part->inertia[0][0] + partMass * (distCenterPart[1]*distCenterPart[1] + distCenterPart[2]*distCenterPart[2])
	// inertia.elements[0][1] = inertia.elements[1][0] -= part->inertia[0][1] - partMass * distCenterPart[0]*distCenterPart[1];
	// and so on.
	// measuring inertia in a frame of reference of the new CoM:
	// inertia.elements[0][0] -= newMass * (distCenters[1]*distCenters[1] + distCenters[2]*distCenters[2])
	// inertia.elements[0][1] = inertia.elements[1][0] += newMass * distCenters[0]*distCenters[1];
	// and so on.
	// doing everything at the same time:
	float centersSq0 = distCenters[0]*distCenters[0];
	float centersSq1 = distCenters[1]*distCenters[1];
	float centersSq2 = distCenters[2]*distCenters[2];
	float partSq0 = distCenterPart[0]*distCenterPart[0];
	float partSq1 = distCenterPart[1]*distCenterPart[1];
	float partSq2 = distCenterPart[2]*distCenterPart[2];

	Matrix3x3 tmp_inertia(currentRigid->inertia());
	tmp_inertia.elements[0][0] -= newMass * (centersSq1 + centersSq2) + part->inertia().elements[0][0] + partMass * (partSq1 + partSq2);
	tmp_inertia.elements[1][1] -= newMass * (centersSq0 + centersSq2) + part->inertia().elements[1][1] + partMass * (partSq0 + partSq2);
	tmp_inertia.elements[2][2] -= newMass * (centersSq1 + centersSq0) + part->inertia().elements[2][2] + partMass * (partSq1 + partSq0);
	tmp_inertia.elements[0][1] += newMass * distCenters[0]*distCenters[1] - part->inertia().elements[0][1] + partMass * distCenterPart[0]*distCenterPart[1];
	tmp_inertia.elements[0][2] += newMass * distCenters[0]*distCenters[2] - part->inertia().elements[0][2] + partMass * distCenterPart[0]*distCenterPart[2];
	tmp_inertia.elements[1][2] += newMass * distCenters[1]*distCenters[2] - part->inertia().elements[1][2] + partMass * distCenterPart[1]*distCenterPart[2];
	tmp_inertia.elements[1][0] = tmp_inertia.elements[0][1];
	tmp_inertia.elements[2][0] = tmp_inertia.elements[0][2];
	tmp_inertia.elements[2][1] = tmp_inertia.elements[1][2];
	currentRigid->set_inertia(tmp_inertia);

	// adapt linear velocity
	const vmml::Vector3f vel = (currentRigid->velocity()*currentRigid->mass() - part->velocity() * partMass) / newMass;
	currentRigid->set_velocity(vel); 

	// adapt angular momentum:
	// remove the angular momentum of part:
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() - (part->angular_momentum() + distCenterPart.cross(part->velocity()*partMass)));
	// move the frame of reference
	currentRigid->set_angular_momentum(currentRigid->angular_momentum() - distCenters.cross(currentRigid->velocity()*newMass));

	currentRigid->set_rigidbody_pos(newCenterOfMass);
	currentRigid->set_mass(newMass);
}

//--------------------------------------------------------------------
void UnifiedPhysics::MarkMadeSolid(const int startIndex, const int endIndex)
{
	for (int i = startIndex; i < endIndex; ++i)
	{
		particles_[i].has_made_solid_ = true;
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::CalculateKernelConstants()
{
	float h = fc_->globalSupportRadius;
	kernel_density_const_ = 315.0f/(64.0f * (float)kPi * h*h*h*h*h*h*h*h*h);	
	kernel_pressure_const_ = -45.f/((float(kPi)*h*h*h*h*h*h));
	kernel_viscosity_const_ = 45.f/((float(kPi)*h*h*h*h*h*h));
	kernel_surface_tension_const_ = - 945.f/(32.f *float(kPi)*pow(h,9));
	kernel_elastic_const_ = -kPi / (8*h*h*h*h*(kPi/3-8/kPi+16/(kPi*kPi)));

	clap_self_ = -945.f*3.f/(32.f*float(kPi)*pow(h,5));

	support_radius_sq_ = h*h;
}

//--------------------------------------------------------------------
float UnifiedPhysics::KernelDensity(float distSq)
{
	// NOTE: This method assumes that |rvec| < fc.support_radius !!
	// Normalized 6th order spline: W(r,h) = 315/(64*pi*h^9)*(h^2 - r^2)^3;

	float f = support_radius_sq_ - distSq;
	return kernel_density_const_ * f * f * f;
}

//--------------------------------------------------------------------
float UnifiedPhysics::KernelPressureGrad(float dist)
{
	// NOTE: This method assumes that |rvec| < fc->globalSupportRadius !!
	// W(r,h) = 15/(pi*h^6)*(h-|r|)^3
	// grad W(r,h) = -45 / (pi*h^6*|r|) * (h-|r|)^2 * r
	// returns only the factor that r has to be scaled to become the gradient

	return kernel_pressure_const_ / dist * (fc_->globalSupportRadius-dist)*(fc_->globalSupportRadius-dist);
}

//--------------------------------------------------------------------
float UnifiedPhysics::KernelViscosityLaplacian(float dist)
{
	// NOTE: This method assumes that |rvec| < fc.support_radius !!
	// lap W(r,h) = -45 / (pi*h^6) * (h - |r|)	

	return kernel_viscosity_const_ * (fc_->globalSupportRadius - dist);
}

// -----------------------------------------------------------------------------------------------
void UnifiedPhysics::KernelSurfaceTension(float distSq, float& grad, float &lap)
{
	// NOTE: This method assumes that |rvec| < fc.support_radius !!
	// using the poly6 spline kernel for the color field:
	// W(r,h) = 315/(64*pi*h^9)*(h^2 - |r|^2)^3;
	// grad W = -945/(32*pi*h^9)*(h^2 - |r|^2)^2 * r   (?)
	// lap W  = -945/(32*pi*h^9)*(|r|^2-h^2)*(7*r^2 - 3*h^2)   (?)

	// grad is set to the factor that r has to be multiplied to get the gradient

	// calculate gradient
	float r2mh2 = distSq - support_radius_sq_;
	float tmp = kernel_surface_tension_const_ * r2mh2;
	grad = tmp * r2mh2;

	// calculate laplacian
	lap = tmp * (7*distSq - 3*support_radius_sq_);
}

//--------------------------------------------------------------------
float UnifiedPhysics::KernelElasticGrad(float dist)
{
	// NOTE: This method assumes that |rvec| < fc->globalSupportRadius !!
	// returns only the factor that r has to be scaled to become the gradient
	static float divisor = kPi/(2*fc_->globalSupportRadius);
	return kernel_elastic_const_*sin((dist+fc_->globalSupportRadius)*divisor)/dist;
}

//--------------------------------------------------------------------
void UnifiedPhysics::UpdateNeighborsRangeParallel(const int startIndex, const int endIndex)
{
	int i;
	int chunk = 100;
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait  	
	for (i = startIndex; i < endIndex; i++)
		GetNeighbors(i);	
}

//--------------------------------------------------------------------
Matrix3x3 UnifiedPhysics::ComputeLocalInertiaTensorRigidBody(const RigidBody& body)
	//--------------------------------------------------------------------
{
	// see "Game Physics" Second edition P57 Eq. (2.81) (2.83) & (2.85)
	Matrix3x3 inertiaTensor;
	inertiaTensor.SetZero();

	const vmml::Vector3f& center_mass = body.rigidbody_pos();
	const std::vector<int>& indices = body.rigid_particle_indices();
	const int num_rigid_particles = indices.size();
	for (int i = 0; i < num_rigid_particles; ++i)
	{
		UnifiedParticle& p = particles_[i];
		vmml::Vector3f relativePos = p.position_ - center_mass;
		float x = relativePos.x;
		float y = relativePos.y;
		float z = relativePos.z;

		inertiaTensor.elements[0][0] += y*y + z*z;
		inertiaTensor.elements[0][1] -= x*y;
		inertiaTensor.elements[0][2] -= x*z;
		inertiaTensor.elements[1][1] += x*x + z*z;
		inertiaTensor.elements[1][2] -= y*z;
		inertiaTensor.elements[2][2] += x*x + y*y;
	}

	inertiaTensor.elements[1][0] = inertiaTensor.elements[0][1];
	inertiaTensor.elements[2][0] = inertiaTensor.elements[0][2];
	inertiaTensor.elements[2][1] = inertiaTensor.elements[1][2];

	inertiaTensor = inertiaTensor * fc_->initialMass;
	return inertiaTensor;
}

//--------------------------------------------------------------------
vmml::Vector3f UnifiedPhysics::ComputeCenterOfMass(std::vector<UnifiedParticle>& particles_)
{
	// centerOfMass = Sum(p_mass * p_position) / num_particles * p_mass = Sum(p_position) / num_particles
	const int num_positions = particles_.size();
	vmml::Vector3f centerOfMass(0.0, 0.0, 0.0);
	for (int i = 0; i < num_positions; ++i)
	{
		centerOfMass += particles_[i].position_;
	}

	centerOfMass /= (float)num_positions;
	return centerOfMass;
}

//--------------------------------------------------------------------
uint UnifiedPhysics::CreateVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

//--------------------------------------------------------------------
void UnifiedPhysics::DoCudaGLInteraction()
{
	// allocate host storage (for VBO purposes)
	const unsigned int totalNumParticles = particles_.size();
	unsigned int vboMemSize = sizeof(float) * 4 * totalNumParticles;

	h_positions_ = new float[4 * totalNumParticles];
	for (uint i = 0; i < totalNumParticles; ++i)
	{
		UnifiedParticle &p = particles_[i];
		h_positions_[i*4]   = p.position_[0];
		h_positions_[i*4+1] = p.position_[1];
		h_positions_[i*4+2] = p.position_[2];
		h_positions_[i*4+3] = p.index_;
	}

	// create vertex buffer objects & reserve space for transferring & storing results
	for (int i = 0; i < 2; ++i)
	{
		dptr_.m_posVbo[i] = CreateVBO(vboMemSize);
		RegisterGLBufferObject(dptr_.m_posVbo[i], &cuda_posvbo_resource_[i]);
	}

	m_color_vbo_ = CreateVBO(vboMemSize);
	RegisterGLBufferObject(m_color_vbo_, &cuda_colorvbo_resource_);
	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_color_vbo_);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for (uint i=0; i<totalNumParticles; i++)
	{
		float t = i / (float) totalNumParticles;
#if 0
		// refer to online RGB color selection tools http://www.atool.org/colorpicker.php 
		
		/*
		const int R = 222;
		const int G = 64;
		const int B = 80;
		*/

		const int R = 30;
		const int G = 144;
		const int B = 255;

		/* purple
		const int R = 227;
		const int G = 25;
		const int B = 230;
		*/
		const float denominator = 255.0f;

		*ptr++ = float(R)/denominator;
		*ptr++ = float(G)/denominator;
		*ptr++ = float(B)/denominator;
#else
		ColorRamp(t, ptr);
		ptr+=3;
#endif
		*ptr++ = 1.0f;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	// transfer data from host to device using OpenGL's vbo function
	// we only need to copy date to dptr_.m_posVbo[0], this would be the input data when mapping with CUDA 
	UnregisterGLBufferObject(cuda_posvbo_resource_[0]);
	glBindBuffer(GL_ARRAY_BUFFER, dptr_.m_posVbo[0]);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vboMemSize, h_positions_);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	RegisterGLBufferObject(dptr_.m_posVbo[0], &cuda_posvbo_resource_[0]);
}

//--------------------------------------------------------------------
void UnifiedPhysics::GlVBOInit()
{
	// allocate host storage (for VBO purposes)
	const unsigned int totalNumParticles = particles_.size();
	unsigned int vboMemSize = sizeof(float) * 4 * totalNumParticles;

	h_positions_ = new float[4 * totalNumParticles];
	for (uint i = 0; i < totalNumParticles; ++i)
	{
		UnifiedParticle &p = particles_[i];
		h_positions_[i*4]   = p.position_[0];
		h_positions_[i*4+1] = p.position_[1];
		h_positions_[i*4+2] = p.position_[2];
		h_positions_[i*4+3] = p.index_;
	}

	m_pos_vbo_cpu_ = CreateVBO(vboMemSize);
	m_color_vbo_ = CreateVBO(vboMemSize);

	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_color_vbo_);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for (uint i=0; i<totalNumParticles; i++)
	{
		float t = i / (float) totalNumParticles;
#if 0
		// refer to online RGB color selection tools http://www.atool.org/colorpicker.php 

		/*
		const int R = 227;
		const int G = 25;
		const int B = 230;
		*/
		
		const float denominator = 255.0f;

		const int R = 30;
		const int G = 144;
		const int B = 255;

		*ptr++ = float(R)/denominator;
		*ptr++ = float(G)/denominator;
		*ptr++ = float(B)/denominator;
#else
		ColorRamp(t, ptr);	
		ptr+=3;
#endif
		*ptr++ = 1.0f;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	// transfer data from host to device using OpenGL's vbo function
	// we need to copy updated positions to m_posVBOCPU after every physics step, Now we just use glBufferSubData to init vbo once
	// later we will use glMapBuffer to update its data
	glBindBuffer(GL_ARRAY_BUFFER, m_pos_vbo_cpu_);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vboMemSize, h_positions_);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

//--------------------------------------------------------------------
void UnifiedPhysics::GlVBOInitWithoutFrozenParticles()
{
	// allocate host storage (for VBO purposes) 
	unsigned int vboMemSize = sizeof(float) * 4 * num_nonfrozen_particles_;
	h_pos_without_frozen_ = new float[4 * num_nonfrozen_particles_];

	const unsigned int totalNumParticles = particles_.size();
	int j = 0;
	for ( uint i = 0; i < totalNumParticles; ++i)
	{
		UnifiedParticle& p = particles_[i];
		if (p.type_ != FROZEN_PARTICLE && j < num_nonfrozen_particles_)
		{
			// is set so to make sure incorrect position values pollute hPosWithoutFrozen array
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;
			float w = 0.0f;

			x = p.position_[0];
			y = p.position_[1];
			z = p.position_[2];
			w = p.index_;

			// we only copy non frozen particle's position to hPosWithoutFrozen
			h_pos_without_frozen_[j*4]   = x;
			h_pos_without_frozen_[j*4+1] = y;
			h_pos_without_frozen_[j*4+2] = z;
			h_pos_without_frozen_[j*4+3] = w;

			++j;
		}	
	}
	
	m_pos_vbo_cpu_ = CreateVBO(vboMemSize);
	m_color_vbo_ = CreateVBO(vboMemSize);

	// fill color buffer
	glBindBufferARB(GL_ARRAY_BUFFER, m_color_vbo_);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for (uint i=0; i<num_nonfrozen_particles_; i++)
	{
		float t = i / (float) num_nonfrozen_particles_;
#if 1
		/*
		*ptr++ = rand() / (float) RAND_MAX;
		*ptr++ = rand() / (float) RAND_MAX;
		*ptr++ = rand() / (float) RAND_MAX;
		//*/

		*ptr++ = 0.0f;
		*ptr++ = 1.0f;
		*ptr++ = 0.5f;
#else
		ColorRamp(t, ptr);
		ptr+=3;
#endif
		*ptr++ = 1.0f;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	// transfer data from host to device using OpenGL's vbo function
	// we need to copy updated positions to m_posVBOCPU after every physics step, Now we just use glBufferSubData to init vbo once
	// later we will use glMapBuffer to update its data
	glBindBuffer(GL_ARRAY_BUFFER, m_pos_vbo_cpu_);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vboMemSize, h_pos_without_frozen_);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

//--------------------------------------------------------------------
void UnifiedPhysics::UpdatePositionVBO()
{
	// update hPositions
	const unsigned int totalNumParticles = particles_.size();
	unsigned int vboMemSize = sizeof(float) * 4 * totalNumParticles;

	for (uint i = 0; i < totalNumParticles; ++i)
	{
		UnifiedParticle &p = particles_[i];
		h_positions_[i*4]   = p.position_[0];
		h_positions_[i*4+1] = p.position_[1];
		h_positions_[i*4+2] = p.position_[2];
		h_positions_[i*4+3] = p.index_;
	}

	// fill pos_vbo_cpu buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_pos_vbo_cpu_);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vboMemSize, h_positions_);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//--------------------------------------------------------------------
void UnifiedPhysics::UpdatePositionVBOWithoutFrozenParticles()
{
	// update hPosWithoutFrozen
	unsigned int vboMemSize = sizeof(float) * 4 * num_nonfrozen_particles_;
	const unsigned int totalNumParticles = particles_.size();

	int j = 0;
	for (uint i = 0; i < totalNumParticles; ++i)
	{
		UnifiedParticle &p = particles_[i];
		if (p.type_ != FROZEN_PARTICLE  && j < num_nonfrozen_particles_)
		{
			// is set so to make sure incorrect position values pollute hPosWithoutFrozen array
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;
			float w = 0.0f;

			x = p.position_[0];
			y = p.position_[1];
			z = p.position_[2];
			w = p.index_;

			// we only copy the updated non frozen particle's position to hPosWithoutFrozen
			h_pos_without_frozen_[j*4]   = x;
			h_pos_without_frozen_[j*4+1] = y;
			h_pos_without_frozen_[j*4+2] = z;
			h_pos_without_frozen_[j*4+3] = w;

			++j;
		}
	}

	// fill pos_vbo_cpu buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_pos_vbo_cpu_);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vboMemSize, h_pos_without_frozen_);	// TODO: As we update all particle positions, will be using glMapBuffer instead
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
}

//--------------------------------------------------------------------
void UnifiedPhysics::HeatingFluid(const int i)
{
	UnifiedParticle &p = particles_[i];
	if((p.position_[1] < 0.03) && (p.type_ != FROZEN_PARTICLE))
	{
		p.temperature_ += 5.0; 
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::CalculateInvertedMomentMatrix(const int pindex)
{
	// Calculates A^-1 for MLS ("Point Based Animation of Elastic, Plastic and Melting Objects")
	// not possible with less than 3 solidNeighs or if these are co-linear/co-planar; because of low precision the determinant will rarely be 0 though, so *some* result is computed

	UnifiedParticle &currentParticle = particles_[pindex];
	// sum_j (x_ij * x_ij [transpose] * w_ij)
	float momentMatrix[3][3];
	memset(momentMatrix, 0, 9*sizeof(float));

	// iterate over solidNeighs
	std::vector<vmml::Vector3f> &psolidNeighDistances = solid_neigh_distances_[pindex];
	const unsigned int nofSolidNeighs = psolidNeighDistances.size(); 
	for(int i = 0; i < nofSolidNeighs; ++i)
	{
		vmml::Vector3f distVec = psolidNeighDistances[i] * -1;	// x_ij = - x_ji WHY: multiply by (-1)?
		float kernel = KernelDensity(distVec.lengthSquared());	// w_ij
		for(int i=0;i<3;++i)
		{
			for(int j=0;j<3;++j)
			{
				momentMatrix[i][j] += distVec[i]*distVec[j]*kernel;
			}
		}
	}// end over all solidNeighs

	// invert momentMatrix
	// -> calculate det(momentMatrix)
	float det =	momentMatrix[0][0]*momentMatrix[1][1]*momentMatrix[2][2] - momentMatrix[2][0]*momentMatrix[1][1]*momentMatrix[0][2]
	+momentMatrix[0][1]*momentMatrix[1][2]*momentMatrix[2][0] - momentMatrix[2][1]*momentMatrix[1][2]*momentMatrix[0][0]
	+momentMatrix[0][2]*momentMatrix[1][0]*momentMatrix[2][1] - momentMatrix[2][2]*momentMatrix[1][0]*momentMatrix[0][1];
	if(det==0)
	{
		std::cout << "Inverse does not exist!" << std::endl;
		currentParticle.strain_factor_.set(0.0f, 0.0f, 0.0f);
		return;
	}
	float detFactor = 1/det;

	// -> calculate inverse by calculating the adjoint ("Komplementäre Matrize") and dividing by det
	for(int i=0; i<3; ++i)
	{
		for(int j=0; j<3; ++j)
		{
			// find 2x2 matrix not containing row i & column j
			float elementsOfSubmatrix[4];
			int elementIndex=0;
			for(int k=0; k<3; ++k)
			{
				for(int l=0; l<3; ++l)
				{
					if(i!=k && j!=l)
					{
						elementsOfSubmatrix[elementIndex] = momentMatrix[k][l];
						++elementIndex;
					}
				}
			}
			currentParticle.inverted_moment_matrix_[j][i] = elementsOfSubmatrix[0]*elementsOfSubmatrix[3]-elementsOfSubmatrix[2]*elementsOfSubmatrix[1];
			// multiply by (-1)^(i+j) to get the adjoint, divide by det to get the inverse:
			currentParticle.inverted_moment_matrix_[j][i] *= ((i+j)&0x1) ? -detFactor : detFactor;
		}
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::UpdateStrain(UnifiedParticle* p, float* elasticStrain)
{
	float avg = (elasticStrain[0] + elasticStrain[1] + elasticStrain[2]) / 3;
	float elasticStrainDeviation[6] = {elasticStrain[0] - avg, elasticStrain[1] - avg, elasticStrain[2] - avg, elasticStrain[3], elasticStrain[4], elasticStrain[5]};
	float frobeniusNorm = 0;
	for(int i=0; i<3; ++i) // frobeniusNorm(e) = sqrt(sum{i} sum {j} e_ij^2)
	{
		// (elasticStrainDeviation[3] = elasticStrainDeviationXY = elasticStrainDeviationYX)
		frobeniusNorm += elasticStrainDeviation[i]*elasticStrainDeviation[i] + 2*elasticStrainDeviation[i+3]*elasticStrainDeviation[i+3];
	}
	frobeniusNorm = sqrt(frobeniusNorm);
	if(frobeniusNorm > p->elastic_limit_) // adapt plastic strain (use equations (Eq. 4.23) to (Eq. 4.26).) from "Simulation of Fluid-Solid Interaction"
	{
		float factor = 1 - p->elastic_limit_/frobeniusNorm;

		for(int i=0; i<6; ++i) 
		{
			p->plastic_strain_[i] += factor * elasticStrainDeviation[i];
		}
		// a lit bitter different from (Eq. 4.26) from "Simulation of Fluid-Solid Interaction" or Eq.(5) from "Graphical Modeling and Animation of Ductile Fracture"
		// should be figured out 
		float newFrobeniusNorm = 0;
		for(int i=0; i<3; ++i) // frobeniusNorm(e) = sqrt(sum{i} sum {j} e_ij^2)
		{
			newFrobeniusNorm += p->plastic_strain_[i] * p->plastic_strain_[i] + 2 * p->plastic_strain_[i+3] * p->plastic_strain_[i+3];
		}
		newFrobeniusNorm = sqrt(newFrobeniusNorm);

		if(newFrobeniusNorm > p->plastic_limit_) // frobeniusNorm(plasticStrain) may not be greater than plasticLimit
		{
			float limitFactor = p->plastic_limit_/newFrobeniusNorm;
			for(int i=0; i<6; ++i) 
			{
				p->plastic_strain_[i] *= limitFactor;
			}
		}
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::UpdateStateOfAggregation()
{
	// iterate over all particles
	const unsigned int particleSize = particles_.size();
	for(int currentParticleIndex = 0; currentParticleIndex < particleSize; ++currentParticleIndex)
	{
		UnifiedParticle &currentParticle = particles_[currentParticleIndex];
		if(currentParticle.type_ == FROZEN_PARTICLE) 
			continue;

		if(currentParticle.temperature_ > 0.5) 
			currentParticle.temperature_ -= fc_->automaticCooling;

		UnifiedParticle::Properties newProperties;
		if(currentParticle.temperature_ <= currentParticle.solid_properties_.temperature) // use solid properties
		{
			newProperties = currentParticle.solid_properties_;
		}
		else if(currentParticle.temperature_ >= currentParticle.fluid_properties_.temperature) // use fluid properties
		{
			newProperties = currentParticle.fluid_properties_;
		}
		else  // interpolate
		{
#ifdef USE_ASSERT
			assert((currentParticle.fluid_properties_.temperature - currentParticle.solid_properties_.temperature) != 0);
#endif
			float factor = (currentParticle.temperature_ - currentParticle.solid_properties_.temperature) / (currentParticle.fluid_properties_.temperature - currentParticle.solid_properties_.temperature);
			newProperties = currentParticle.solid_properties_ + (currentParticle.fluid_properties_ - currentParticle.solid_properties_) * factor;
		}
		// set these properties
		currentParticle.young_modulus_ = newProperties.young_modulus;
		currentParticle.visc_const_ = newProperties.visc_const;

		// make new connections when freezing
		if(currentParticle.old_temperature_>currentParticle.solid_properties_.temperature && currentParticle.temperature_<=currentParticle.solid_properties_.temperature)
		{
			FreezeParticle(currentParticleIndex);
		}

		// get non-rigid when too warm
		if(currentParticle.parent_rigidbody_ && currentParticle.old_temperature_<=currentParticle.solid_properties_.temperature && currentParticle.temperature_>currentParticle.solid_properties_.temperature)
		{
			SplitRigidBody(currentParticleIndex);
		}

		// split connections to neighbors if:
		if(currentParticle.young_modulus_ <= 0 && !solid_neighs_.empty() && !solid_neighs_[currentParticleIndex].empty())
		{
			MeltParticle(currentParticleIndex);
		}

		currentParticle.CalculateHookeMatrix();
		currentParticle.old_temperature_ = currentParticle.temperature_;
	}// end over all particles
}

// Note : this function could not be parallelized straightforward because of solidNeighs modification
// solidNeighs[currentParticleIndex].push_back(neighIndex);
// solidNeighs[neighIndex].push_back(currentParticleIndex);	// TODO: need to adapted for parallelization
//--------------------------------------------------------------------
void UnifiedPhysics::FreezeParticle(const int currentParticleIndex)
{
	UnifiedParticle *currentParticle = &particles_[currentParticleIndex];
	bool rigid = currentParticle->rigidbody_; // currentParticle becomes part of a rigid object
	std::vector<RigidBody*> objectsToMerge; // will contain objects that are to be combined with each other and currentParticle
	std::vector<int> particlesToMerge; // will contain particles that are to be combined to an object
	if(rigid)
	{
		if(currentParticle->parent_rigidbody_)
		{
			objectsToMerge.push_back(currentParticle->parent_rigidbody_);
		}
		else
		{
			particlesToMerge.push_back(currentParticleIndex);
		}
		// look for solidNeighs that should belong to the same object
		int solidNeighSize;
		std::vector<int> &pSolidNeighs = solid_neighs_[currentParticleIndex];
		if (solid_neighs_.empty())
		{
			solidNeighSize = 0;
		}
		else
		{
			solidNeighSize = pSolidNeighs.size();
		}
		
		for(int i = 0; i < solidNeighSize; ++i)
		{
			int solidNeighIndex = pSolidNeighs[i];
			
			UnifiedParticle* solidNeigh = &particles_[solidNeighIndex];
			if(solidNeigh->rigidbody_ && solidNeigh->temperature_ < solidNeigh->solid_properties_.temperature)
			{
				// solidNeigh should belong to the same object
				if(solidNeigh->parent_rigidbody_)
				{
					objectsToMerge.push_back(solidNeigh->parent_rigidbody_);
				}
				else
				{
					particlesToMerge.push_back(solidNeighIndex);
				}
			}
		}
	}

	int neighSize;
	std::vector<int> &pNeighs = neighbor_indices_[currentParticleIndex];
	if (neighbor_indices_.empty())
	{
		neighSize = 0;
	}
	else
	{
		neighSize = pNeighs.size();
	}
	
	int solidNeighSize;
	if (solid_neighs_.empty())
	{
		solidNeighSize = 0;
	}
	else
	{
		solidNeighSize = solid_neighs_[currentParticleIndex].size();
	}
	
	for(int i = 0; i < neighSize; ++i)
	{
		int neighIndex = pNeighs[i];
		
		UnifiedParticle* neigh = &particles_[neighIndex];

		if(neigh->temperature_ < neigh->solid_properties_.temperature && neigh->type_ != FROZEN_PARTICLE) // neighbour is 'frozen', too
		{
			// actually, particle might already have solidNeighs, and neigh might be one of them
			// (if it has frozen, warmed up a little and is freezing again)
			bool shouldAdd = true;
			for(int i=0; i<solidNeighSize; ++i)  // setting solidNeighSize only once prevents us from searching neighs we added in this function call (e.g. if solidNeighs was empty in the beginning, we will never search this vector)
			{
				std::vector<int> &solidNeighParticles = solid_neighs_[currentParticleIndex];
				if(solidNeighParticles[i] == neighIndex)
				{
					shouldAdd = false;
					break;
				}
			}
			if(shouldAdd)
			{
				vmml::Vector3f distVec = currentParticle->position_ - neigh->position_;

				// update solidVolume
				float kernel = KernelDensity(distVec.lengthSquared());
				currentParticle->solid_volume_=currentParticle->particle_mass_/(currentParticle->particle_mass_/currentParticle->solid_volume_ + kernel*neigh->particle_mass_);  // (Eq. 6.4) from "Simulation of Fluid-Solid Interaction"
				neigh->solid_volume_=neigh->particle_mass_/(neigh->particle_mass_/neigh->solid_volume_ + kernel*currentParticle->particle_mass_);	// TODO: need to adapted for parallelization

#ifdef USE_ASSERT
				assert(!solid_neighs_.empty());
#endif

				solid_neighs_[currentParticleIndex].push_back(neighIndex);
				solid_neighs_[neighIndex].push_back(currentParticleIndex);	// TODO: need to adapted for parallelization

#ifdef USE_ASSERT
				assert(!solid_neigh_distances_.empty());
#endif

				solid_neigh_distances_[currentParticleIndex].push_back(distVec);			
				solid_neigh_distances_[neighIndex].push_back(distVec*-1);	// TODO: need to adapted for parallelization

				if(rigid && neigh->rigidbody_)
				{
					// neigh should belong to the same object
					if(neigh->parent_rigidbody_)
					{
						objectsToMerge.push_back(neigh->parent_rigidbody_);
					}
					else
					{
						particlesToMerge.push_back(neighIndex);
					}
				}
			}
		}
	}
	// add objects
	if(rigid)
	{
		RigidBody* combinedBody = NULL; // this will point to the object that contains currentParticle
		// don't add objects twice! this would happen if two neighbours of currentParticle belong
		// to the same object. sorting will make ignoring copies easier
		std::sort(objectsToMerge.begin(),objectsToMerge.end());

		// if there are some objects, combine them
		if(objectsToMerge.size()>0)
		{
			combinedBody = objectsToMerge[0];
			RigidBody* lastAddedBody = combinedBody;
			for(int i=1; i<objectsToMerge.size(); ++i)
			{
				RigidBody* part = objectsToMerge[i];
				if(part != lastAddedBody) // only combine if 'part' has not been added yet
				{
					AddRigidBodyParticles(combinedBody, part);
					// remove part from the list of objects in the simulation
					std::vector<RigidBody*>::iterator endIter = rigid_bodies_.end();
					for(std::vector<RigidBody*>::iterator rb=rigid_bodies_.begin(); rb!=endIter; ++rb)
					{
						if(*rb == part)
						{
							rigid_bodies_.erase(rb);
							break;
						}
					}
					delete part;
					lastAddedBody = part;
				}
			}
			if(particlesToMerge.size()>0)
			{
				std::vector<int>::iterator endIter = particlesToMerge.end();
				for(std::vector<int>::iterator n = particlesToMerge.begin(); n != endIter; ++n)
				{
					UnifiedParticle* part = &particles_[*n];
					AddToRigidBody(combinedBody, *n);
				}
			}
		}
		else if(particlesToMerge.size()>1) // if there are only particles (and more than just currentParticle), make a new object
		{
			combinedBody = new RigidBody();
			std::vector<int>::iterator endIter = particlesToMerge.end();
			for(std::vector<int>::iterator n = particlesToMerge.begin(); n != endIter; ++n)
			{
				UnifiedParticle* part = &particles_[*n];
				AddToRigidBody(combinedBody, *n);
			}
			rigid_bodies_.push_back(combinedBody);
		}
		// invert inertia if currentParticle could be added somewhere
		if(combinedBody)
		{
			combinedBody->InvertInertia();
		}
		// (if there are neither objectsToMerge nor particlesToMerge, do nothing)
	}

	//CalculateInvertedMomentMatrix(currentParticle);
}

//--------------------------------------------------------------------
void UnifiedPhysics::MeltParticle(const int pIndex)
{
	// remove currentParticle from every solid neighbour particle's solidNeighs
	UnifiedParticle& currentParticle = particles_[pIndex];
	
	// iterate over all solid neighs
	std::vector<int>& psolidNeighs = solid_neighs_[pIndex];
	const unsigned int nofpSolidNeighs = psolidNeighs.size();
	for(int i = 0; i < nofpSolidNeighs; ++i)
	{
		const int neighIndex = psolidNeighs[i];
		UnifiedParticle &neigh = particles_[neighIndex];; 
		
		// iterate over solidNeighs of currentParticle's neighbour particle 
		// delete the currentParticle
		// the solidDistance of the neighbour is deleted at the same time

		std::vector<int>::iterator endSearch = solid_neighs_[neighIndex].end();
		std::vector<vmml::Vector3f>::iterator distance = solid_neigh_distances_[neighIndex].begin();
		int neighIndexOfNeigh = 0; 
		for(std::vector<int>::iterator nn = solid_neighs_[neighIndex].begin(); nn != endSearch; ++nn, ++distance, ++neighIndexOfNeigh)
		{
			if(neighIndexOfNeigh == pIndex)
			{
				vmml::Vector3f distVec = *distance;
				float kernel = KernelDensity(distVec.lengthSquared());
				neigh.solid_volume_ = neigh.particle_mass_/(neigh.particle_mass_/neigh.solid_volume_ - kernel*currentParticle.particle_mass_);

				solid_neighs_[neighIndex].erase(nn);
				solid_neigh_distances_[neighIndex].erase(distance);
				break;
			}
		}// end over neighbour particle's solidNeighs
	}// end over solid neighs
	currentParticle.solid_volume_ = 1/kernel_self_; // use kernelSelf to replace KernelDensity(0)
	// remove all solidNeighs
	solid_neighs_[pIndex].clear();
	solid_neigh_distances_[pIndex].clear();
	memset(currentParticle.plastic_strain_, 0, 6*sizeof(float)); // forget about plastic strain
}

//--------------------------------------------------------------------
void UnifiedPhysics::SplitRigidBody(const int linkingParticleIndex)
{
	UnifiedParticle &linkingParticle = particles_[linkingParticleIndex];
	RigidBody* parentBody = linkingParticle.parent_rigidbody_;
	// mark all particles in parentBody unparsed 
	const unsigned int rigidBodyParticlesSize = parentBody->rigid_particle_indices().size();
	for(int pIndex = 0; pIndex < rigidBodyParticlesSize; ++pIndex)
	{
		particles_[pIndex].has_been_parsed_ = false;
	}
	RemoveFromRigidBody(parentBody, &linkingParticle);
	linkingParticle.parent_rigidbody_ = NULL;
	linkingParticle.has_been_parsed_ = true;
	//vector<UnifiedParticle*> unvisitedLinkNeighs = linkingParticleIndex->solidNeighs; // copy a list of direct neighbours of linkingParticleIndex. if it is empty, the iteration can be stopped immediately
	bool parentComplete = false; // the first set of particles goes to parentBody, the rest to new objects
	// recurse over the neighbours to see if splitting is necessary
	std::vector<int> &linkingParticleNeighs = solid_neighs_[linkingParticleIndex];
	const unsigned int linkingParticleNeighsSize = linkingParticleNeighs.size();
	for(int linkNeighIndex = 0; linkNeighIndex < linkingParticleNeighsSize; ++linkNeighIndex)
	{
		UnifiedParticle &linkNeighParticle = particles_[linkNeighIndex];
		// if a solid neighbour has a parentRigidBody, it is the same as parentBody
		if(!linkNeighParticle.has_been_parsed_ && linkNeighParticle.parent_rigidbody_)
		{
			linkNeighParticle.has_been_parsed_ = true;

			std::vector<int> particlesOfCurrentBody; // holds all particles that belong to 1 rigid body
			particlesOfCurrentBody.push_back(linkNeighIndex);

			SplitRecursively(linkNeighIndex, particlesOfCurrentBody);

			if(particlesOfCurrentBody.size()<2) // no particle was added -> linkNeighIndex is not part of an object any more
			{
				RemoveFromRigidBody(parentBody, &linkNeighParticle);
				linkNeighParticle.parent_rigidbody_ = NULL;
			}
			else if(parentComplete)
			{
				// add the particles to a new object
				RigidBody* newBody = new RigidBody();
				rigid_bodies_.push_back(newBody);
				for(std::vector<int>::iterator p = particlesOfCurrentBody.begin(); p != particlesOfCurrentBody.end(); ++p)
				{
					AddToRigidBody(newBody, *p);
				}
				RemoveRigidBodyParticles(parentBody, newBody);
				newBody->InvertInertia();
			}
			else // the original object consists of the particles that have just been parsed
			{
				parentComplete = true;
				parentBody->rigid_particle_indices() = particlesOfCurrentBody;
			}
		}
	}
	if(parentComplete)
	{
		parentBody->InvertInertia();
	}
	else // the rigid body to which linkingParticleIndex belonged consists of less than 2 particles
	{
		// remove parentBody from the rigid body vector
		std::vector<RigidBody*>::iterator endIter = rigid_bodies_.end();
		for(std::vector<RigidBody*>::iterator rb=rigid_bodies_.begin(); rb!=endIter; ++rb)
		{
			if(*rb == parentBody)
			{
				rigid_bodies_.erase(rb);
				break;
			}
		}
		// ...and get rid of it
		delete parentBody;
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::SplitRecursively(const int currentParticleIndex, std::vector<int>& collectedParticles)
{
	const unsigned int solidNeighSize = solid_neighs_[currentParticleIndex].size();
	for(int solidNeighIndex = 0; solidNeighIndex < solidNeighSize; ++solidNeighIndex)
	{
		UnifiedParticle &solidNeigh = particles_[solidNeighIndex];
		if(!solidNeigh.has_been_parsed_ && solidNeigh.parent_rigidbody_)
		{
			solidNeigh.has_been_parsed_ = true;
			// TODO: (remove solidNeighIndex if it's in unvisitedLinkNeighs)
			collectedParticles.push_back(solidNeighIndex);
			SplitRecursively(solidNeighIndex, collectedParticles);
		}
	}
}

void UnifiedPhysics::SetParticleCount(uint32_t count)
{
	particle_count_ = count;
}

vmml::Vector3f UnifiedPhysics::CalculateSurfaceTensionForcePCISPHHost(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);
	const float splineCoefficient = my_kernel_->lut_spline_surface_tension()[dist_lut];

	force = (p_pos - neigh_pos) * fc_->surface_tension_gamma * fc_->initialMass * fc_->initialMass * splineCoefficient * (1.0f/dist);

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateSurfaceCohesionForcePCISPHHost(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, float neigh_weighted_vol)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);
	const float splineCoefficient = my_kernel_->lut_spline_surface_adhesion()[dist_lut];

	force = (p_pos - neigh_pos) * fc_->surface_adhesion_beta * fc_->initialMass * fc_->fluidRestDensity * neigh_weighted_vol * splineCoefficient * (1.0f/dist); 

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateSpringForceHost(const float& dist, const float& overlap, const vmml::Vector3f& r_ij)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);

	force.x = -1.0f * fc_->springCoefficient * overlap * r_ij.x / dist;
	force.y = -1.0f * fc_->springCoefficient * overlap * r_ij.y / dist;
	force.z = -1.0f * fc_->springCoefficient * overlap * r_ij.z / dist;

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateDampingForceHost(const vmml::Vector3f& v_ij)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);

	force.x = fc_->dampingCoefficient * v_ij.x;
	force.y = fc_->dampingCoefficient * v_ij.y;
	force.z = fc_->dampingCoefficient * v_ij.z;

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateBoundaryForcePerLiquidParticleHost(const vmml::Vector3f& position)
{
	vmml::Vector3f f(0.0f, 0.0f, 0.0f);
	const float invforceDistance = 1.0f/fc_->forceDistance;
	const float maxBoundaryForce = fc_->maxBoundaryForce; 

	if( position.x < fc_->collisionBox.getMin().x + fc_->forceDistance )
	{
		f += (vmml::Vector3f(1.0,0.0,0.0) * ((fc_->collisionBox.getMin().x + fc_->forceDistance - position.x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.x > fc_->collisionBox.getMax().x - fc_->forceDistance )
	{
		f += (vmml::Vector3f(-1.0,0.0,0.0) * ((position.x + fc_->forceDistance - fc_->collisionBox.getMax().x) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.y < fc_->collisionBox.getMin().y + fc_->forceDistance )
	{
		f += (vmml::Vector3f(0.0,1.0,0.0) * ((fc_->collisionBox.getMin().y + fc_->forceDistance - position.y) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.y > fc_->collisionBox.getMax().y - fc_->forceDistance )
	{
		f += (vmml::Vector3f(0.0,-1.0,0.0) * ((position.y + fc_->forceDistance - fc_->collisionBox.getMax().y) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.z < fc_->collisionBox.getMin().z + fc_->forceDistance )
	{
		f += (vmml::Vector3f(0.0,0.0,1.0) * ((fc_->collisionBox.getMin().z + fc_->forceDistance - position.z) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	if( position.z > fc_->collisionBox.getMax().z - fc_->forceDistance )
	{
		f += (vmml::Vector3f(0.0,0.0,-1.0) * ((position.z + fc_->forceDistance - fc_->collisionBox.getMax().z) * invforceDistance * 2.0 * maxBoundaryForce));
	}

	return f;
}

vmml::Vector3f UnifiedPhysics::CalculateBoundaryForcePerRigidParticleHost(const vmml::Vector3f& position, const vmml::Vector3f& vel)
{
	vmml::Vector3f f(0.0f, 0.0f, 0.0f);
	bool collisionOccured = false;

	const float particle_radius = fc_->particleRadius;
	const float spring_boundary_coefficient = fc_->springCoefficientBoundary;
	const float damping_coefficient = fc_->dampingCoefficient;
	const vmml::Vector3f& minCollisionBox = fc_->collisionBox.getMin();
	const vmml::Vector3f& maxCollisionBox = fc_->collisionBox.getMax();

	// Left wall
	if( position.x - particle_radius < minCollisionBox.x )
	{
		collisionOccured = true;
		f += vmml::Vector3f(1.0,0.0,0.0) * (minCollisionBox.x + particle_radius - position.x) * spring_boundary_coefficient;
	}

	// Right wall
	if( position.x + particle_radius > maxCollisionBox.x )
	{
		collisionOccured = true;
		f += vmml::Vector3f(-1.0,0.0,0.0) * (position.x + particle_radius - maxCollisionBox.x) * spring_boundary_coefficient;
	}

	// Ground collision
	if( position.y - particle_radius < minCollisionBox.y )
	{
		collisionOccured = true;
		f += vmml::Vector3f(0.0,1.0,0.0) * (minCollisionBox.y + particle_radius - position.y) * spring_boundary_coefficient;
	}

	// Ceil collision
	if( position.y + particle_radius > maxCollisionBox.y )
	{
		collisionOccured = true;
		f += vmml::Vector3f(0.0,-1.0,0.0) * (position.y + particle_radius - maxCollisionBox.y) * spring_boundary_coefficient;
	}

	// Back wall
	if( position.z - particle_radius < minCollisionBox.z )
	{
		collisionOccured = true;
		f += vmml::Vector3f(0.0,0.0,1.0) * (minCollisionBox.z + particle_radius - position.z) * spring_boundary_coefficient;
	}

	// Front wall
	if( position.z + particle_radius > maxCollisionBox.z )
	{
		collisionOccured = true;
		f += vmml::Vector3f(0.0,0.0,-1.0) * (position.z + particle_radius - maxCollisionBox.z) * spring_boundary_coefficient;
	}

	if (collisionOccured)
	{
		f.x -= damping_coefficient * vel.x;
		f.y -= damping_coefficient * vel.y;
		f.z -= damping_coefficient * vel.z;
	}	

	return f;
}

void UnifiedPhysics::CreatePartilcesCPU(float spacing, float jitter, BBox& fluidBox) 
{
	vmml::Vector3f delta = fluidBox.getMax()-fluidBox.getMin();
	vmml::Vector3i numParticles = delta / vmml::Vector3f(spacing);

#ifndef SPH_DEMO_SCENE_2 
	particle_count_ = numParticles.x * numParticles.y * numParticles.z;
	//particles.clear();
	//CreateParticleLayer();
	//particle_count_ = particles.size();
#endif

	fc_->setParticleSpacing(spacing);
	//cout << scales <<  " " << vmml::Vector3f(GRID_RESOLUTION) / (virtualBoundingBox.getMax()-virtualBoundingBox.getMin()) << endl;

	float my_block_size = fc_->scales.x * fc_->globalSupportRadius;
	my_block_size = MAX(my_block_size, fc_->scales.y * fc_->globalSupportRadius); 
	my_block_size = MAX(my_block_size, fc_->scales.z * fc_->globalSupportRadius); 

	/*cout << "Individual block sizes : " << endl;
	cout << fc->scales.x * fc->globalSupportRadius << " " << 
	fc->scales.y * fc->globalSupportRadius << " " << 
	fc->scales.z * fc->globalSupportRadius << " " << endl;*/

	// setup for spatial queries
	zIndex.SetBlockSize(my_block_size);

	// In the new demo scene, particles are created every frame
#ifdef SPH_DEMO_SCENE_2
	CreateParticleLayer();

	double particleVolume = fc_->particleSpacing * fc_->particleSpacing 
		* fc_->particleSpacing;
	fc_->initialMass = fc_->fluidRestDensity * particleVolume;
#else    
	particles_.resize(particle_count_);
	neighbor_indices_.resize(particle_count_);
	int counter = 0;
	for(int iy = 0; iy < numParticles.y; iy++) {
		float y = jitter + iy * spacing;

		for(int ix = 0; ix < numParticles.x; ix++) {
			float x = jitter + ix * spacing;

			for(int iz = 0; iz < numParticles.z; iz++) {	
				float z = jitter + iz * spacing;		

				if(counter > particle_count_)
					continue;
				// jittered position within virtualBoundingBox
				particles_[counter].position_.x = x + drand48() * jitter;
				particles_[counter].position_.y = y + drand48() * jitter; 
				particles_[counter].position_.z = z + drand48() * jitter;

				// get linearized z-index for integer position within virtualBoundingBox
				particles_[counter].index_ = zIndex.CalcIndex(fc_->scales * particles_[counter].position_);

				// set physics attributes
				particles_[counter].velocity_ = (0.0, 0.0, 0.0);

				counter++;
			}
		}
	}
	//assert( counter == particles.size() );

	SortParticles();

	printf("%d particles initialized and sorted according to its z-index.\n", particles_.size());
#endif

}

// create a color ramp
void UnifiedPhysics::ColorRamp(float t, float *r)
{
	const int ncolors = 7;
	float c[ncolors][3] =
	{
		{ 1.0, 0.0, 0.0, },
		{ 1.0, 0.5, 0.0, },
		{ 1.0, 1.0, 0.0, },
		{ 0.0, 1.0, 0.0, },
		{ 0.0, 1.0, 1.0, },
		{ 0.0, 0.0, 1.0, },
		{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors-1);
	int i = (int) t;
	float u = t - floor(t);
	r[0] = Lerp(c[i][0], c[i+1][0], u);
	r[1] = Lerp(c[i][1], c[i+1][1], u);
	r[2] = Lerp(c[i][2], c[i+1][2], u);
}

void UnifiedPhysics::CreateInitParticles(float spacing, float jitter, BBox& fluidBox)
{    
	CreatePartilcesCPU(spacing, jitter, fluidBox);

#ifdef USE_CUDA

	// CUDA stuff
	InitCudaVariables(particle_count_, spacing);

#endif

}

void UnifiedPhysics::GetNeighbors(const int i)
{
	UnifiedParticle &p = particles_[i];

	// set query box
	vmml::Vector3f min_query_box(fc_->scales * (p.position_ - fc_->globalSupportRadius)), max_query_box(fc_->scales * (p.position_ + fc_->globalSupportRadius));
	// limit to bounding box of spatial domain
	for (int i = 0; i < 3; i++) {
		min_query_box[i] = MAX(min_query_box[i], 0);
		max_query_box[i] = MIN(max_query_box[i], GRID_RESOLUTION);
	}

	// get index ranges covering the given query box
	ZRanges ranges;
	zIndex.BoxQuery(min_query_box, max_query_box, ranges);

	// sort ranges
	QuickSort(ranges.begin(), ranges.end());

	// reset numbers of neighbors
	std::vector<int> &neighbor_indices = neighbor_indices_[i];
	neighbor_indices.clear();

	// identify particles within ranges
	std::vector<UnifiedParticle>::iterator q;
	int counter = 0;
	for (int k = 0; k < ranges.size(); k++) {
		// the elements in the range shall already be sorted for lower_bound algorithm
		for (q = std::lower_bound(particles_.begin(), particles_.end(), ranges[k].first); q < particles_.end() && q->index_ <= ranges[k].second; q++) {
			int j = q - particles_.begin();
			if (j == i) continue;

			float dist = p.position_.distance(q->position_);
			if (dist < fc_->globalSupportRadius) {
				counter++;
				neighbor_indices.push_back(j);
			}
		}
	}	

	/*
	// loop over all other particles and check distances 
	counter = 0;
	for (j = 0; j < particles.size(); j++) {
	if(&p == &particles[j]) continue;

	float dist = p.position.distance(particles[j].position);	
	if (dist < fc->globalSupportRadius)  {
	counter++;
	neighbor_indices.push_back(j);
	}
	}
	*/
}

void UnifiedPhysics::ComputePhysics(const int i)
{
	static const vmml::Vector3f minCB = fc_->collisionBox.getMin();
	static const vmml::Vector3f maxCB = fc_->collisionBox.getMax();
	static const vmml::Vector3f gravity_direction(0.0, -1.0, 0.0);
	float force, acc;
	UnifiedParticle &p = particles_[i];

	// physical forces
	force = 9.81 * FluidMass;
	acc = force / FluidMass;

	// update position
	p.velocity_ += gravity_direction * acc;
	p.position_ += p.velocity_ * fc_->deltaT;

	// detect boundary collision
	srand(2013);
	if (p.position_.y < minCB.y) {
		p.position_.y = minCB.y;
		long factor = rand()%10;
		p.velocity_.scale(-(0.8 + 0.01 * factor));
		p.velocity_.x += 1.0*(factor-5);
		factor = rand()%10;
		p.velocity_.z += 1.0*(factor-5);
	}

	for (int j = 0; j < 3; j += 2) {
		if (p.position_[j] < minCB[j]) {
			p.position_[j] = minCB[j];
			p.velocity_[j] *= -0.9;
		}
		if (p.position_[j] > maxCB[j]) {
			p.position_[j] = maxCB[j];
			p.velocity_[j] *= -0.9;
		}
	}

	// get new spatial index
	p.index_ = zIndex.CalcIndex(fc_->scales * p.position_);	
}

void UnifiedPhysics::CalculateDensityPressure(const int i)
{
	// calculate densities only for liquid particles
	UnifiedParticle &p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		std::vector<int> &neighbor_indices = neighbor_indices_[i];

		// add own contribution
		p.density_ = kernel_self_ * fc_->initialMass;

		// iterate over all neighbors
		for (int j = 0; j < neighbor_indices.size(); j++) {
			float dist = p.position_.distance(particles_[neighbor_indices[j]].position_);
			if (dist < fc_->globalSupportRadius) {
				// sum up contribution from neighbors
				float kernelValue = my_kernel_->kernelM4Lut(dist); // symmetric
				p.density_ += kernelValue * fc_->initialMass; // symmetric
			}
		}

		// add wall weight functions
		if (fc_->addWallWeightFunction)
		{
			/* using LUT 
			// TODO: Problem exists
			const float distToWall = GetDistToWallLut(p.position.x, p.position.y, p.position.z);
			const float wallWeightValue = GetWallWeightFunctionLut(distToWall);
			//*/

			//* direct compute
			const float distToWall = DistanceToWall(p.position_.x, p.position_.y, p.position_.z);
			static const float effectiveDist = fc_->globalSupportRadius - fc_->particleRadius;
			const float wallWeightValue = WallWeight(distToWall, effectiveDist);
			//*/

			p.density_ += wallWeightValue;
		}


		// stats / debugging
		if(p.density_ < min_dens_) min_dens_ = p.density_;
		if(p.density_ > max_dens_) max_dens_ = p.density_;	
		dens_avg_timestep_ += p.density_;

		// correct for density near boundary, seems not good enough ***
		/*
		dist = p.position.y - collisionBox.getMin().y;
		if (dist < fc->globalSupportRadius) {
		dist = 0.5 * fc->globalSupportRadius - dist;
		float volume = 4.0/3.0 * M_PI * fc->globalSupportRadius*fc->globalSupportRadius*fc->globalSupportRadius;
		float cap = M_PI/6.0 * (3.0 * (fc->globalSupportRadius*fc->globalSupportRadius - (fc->globalSupportRadius - dist) * (fc->globalSupportRadius - dist)) + dist*dist) * dist;
		float factor = volume / (volume -cap);
		p.density *= factor;
		}
		*/

		// after having accumulated all contributions we can compute the pressure
		if(fc_->physicsType == 'o')
		{
			p.pressure_ = CalculatePressureSPH(p.density_);
		}
		else if(fc_->physicsType == 'w')
		{
			p.pressure_ = CalculatePressureWCSPH(p.density_);
		}
	}
}

void UnifiedPhysics::CalculateDensityPressurePureFluidSPH(const int i)
{
	// calculate densities only for liquid particles
	UnifiedParticle &p = particles_[i];

#ifdef USE_DEBUG
	assert(p.type_ == LIQUID_PARTICLE);
#endif

	std::vector<int> &neighbor_indices = neighbor_indices_[i];

	// add own contribution
	p.density_ = kernel_self_ * fc_->initialMass;

	// iterate over all neighbors
	for (int j = 0; j < neighbor_indices.size(); j++) {
		float dist = p.position_.distance(particles_[neighbor_indices[j]].position_);
		if (dist < fc_->globalSupportRadius) {
			// sum up contribution from neighbors
			float kernelValue = my_kernel_->kernelM4Lut(dist); // symmetric
			p.density_ += kernelValue * fc_->initialMass; // symmetric
		}
	}

	// add wall weight functions
	if (fc_->addWallWeightFunction)
	{
		/* using LUT 
		// TODO: Problem exists
		const float distToWall = GetDistToWallLut(p.position.x, p.position.y, p.position.z);
		const float wallWeightValue = GetWallWeightFunctionLut(distToWall);
		//*/

		//* direct compute
		const float distToWall = DistanceToWall(p.position_.x, p.position_.y, p.position_.z);
		static const float effectiveDist = fc_->globalSupportRadius - fc_->particleRadius;
		const float wallWeightValue = WallWeight(distToWall, effectiveDist);
		//*/

		p.density_ += wallWeightValue;
	}


	// stats / debugging
	if(p.density_ < min_dens_) min_dens_ = p.density_;
	if(p.density_ > max_dens_) max_dens_ = p.density_;	
	dens_avg_timestep_ += p.density_;

	// correct for density near boundary, seems not good enough ***
	/*
	dist = p.position.y - collisionBox.getMin().y;
	if (dist < fc->globalSupportRadius) {
	dist = 0.5 * fc->globalSupportRadius - dist;
	float volume = 4.0/3.0 * M_PI * fc->globalSupportRadius*fc->globalSupportRadius*fc->globalSupportRadius;
	float cap = M_PI/6.0 * (3.0 * (fc->globalSupportRadius*fc->globalSupportRadius - (fc->globalSupportRadius - dist) * (fc->globalSupportRadius - dist)) + dist*dist) * dist;
	float factor = volume / (volume -cap);
	p.density *= factor;
	}
	*/

	// after having accumulated all contributions we can compute the pressure
	if(fc_->physicsType == 'o')
	{
		p.pressure_ = CalculatePressureSPH(p.density_);
	}
	else if(fc_->physicsType == 'w')
	{
		p.pressure_ = CalculatePressureWCSPH(p.density_);
	}
}

float UnifiedPhysics::CalculatePressureSPH(const float density)
{
	// original SPH
	const float gamma = 1.0;
	const float b_i = fc_->fluidRestDensity * fc_->fluidGasConst / gamma;		
	const float densityFrac = density / fc_->fluidRestDensity;
	const float powGamma = densityFrac;
	float pressure = MAX(0.0f, b_i * (powGamma - 1.0f));	// p = k(rho - rho_0)

	return pressure;
}

float UnifiedPhysics::CalculatePressureWCSPH(const float density)
{
	// WCSPH
	const float gamma = 7.0;
	const float b_i = fc_->fluidRestDensity * fc_->fluidGasConstantWCSPH / gamma;		
	const float densityFrac = density / fc_->fluidRestDensity;
	const float powGamma = pow(densityFrac, gamma);		
	float pressure = MAX(0.0f, b_i * (powGamma - 1.0f));

	return pressure;
}

void UnifiedPhysics::CalculateWeightedVolumeWithshepard_filter_den_kernel_factor(const int i)
{
	UnifiedParticle &p = particles_[i];
	std::vector<int> &neighbor_indices = neighbor_indices_[i];
	
	// calculate corrected kernels "Shepard Filter" according to E.q(2) from paper "Direct forcing for Lagrangian rigid-fluid coupling" for every particle
	float inverse_volume = 0.0f;
	float sum_Wc = 0.0f;
	// iterate over all particle neighbors and accumulate it's inverse volume & sum_Wc if any
	for (int j = 0; j < neighbor_indices.size(); ++j) 
	{
		UnifiedParticle& neigh = particles_[neighbor_indices[j] ];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius)
		{
			// all real neighbors including both liquid & frozen ones
			float kernelValue = my_kernel_->kernelM4Lut(dist);
			sum_Wc += kernelValue;

			// calculate weighted volume of boundary particle according to E.q(4) from paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
			if (p.type_ == FROZEN_PARTICLE && neigh.type_ == FROZEN_PARTICLE)
			{	
				// for real boundary neighbor particles
				inverse_volume += kernelValue;
			}
		}
	}

	// update Shepard filter corrected density kernel factor	
	if (sum_Wc != 0.0)
	{
		p.shepard_filter_den_kernel_factor_ = 1.0f / (sum_Wc * fc_->fluidRestVol);
	}

	// update weighted boundary particle volume
	if (p.type_ == FROZEN_PARTICLE)
	{
		float V_bi = 0.0f;	// boundary particle volume
		if (inverse_volume != 0)
		{
			V_bi = 1.0 / inverse_volume;
		}
		else
		{
			std::cerr << "inverse_volume == 0!!! ERROR" << std::endl;
			abort();
		}
		// update weighted_volume for subsequent usage in boundary pressure force calculation
		// TODO: parallelize this
		p.weighted_volume_ = V_bi;
	}

	/* debugging
	if (p.type == LIQUID_PARTICLE)
	{
		std::cout << "liquid particle : p.shepard_filter_den_kernel_factor = " << p.shepard_filter_den_kernel_factor << std::endl;
		std::cout << "liquid particle : p.weighted_volume = " << p.weighted_volume << std::endl << std::endl;
	}
	else if (p.type == FROZEN_PARTICLE)
	{
		std::cout << "frozen particle : p.shepard_filter_den_kernel_factor = " << p.shepard_filter_den_kernel_factor << std::endl;
		std::cout << "frozen particle : p.weighted_volume = " << p.weighted_volume << std::endl << std::endl;
	}
	//*/
}

void UnifiedPhysics::CalculateCorrectedDensitiesVersatileCouplingWithshepard_filter_den_kernel_factor(const int i)
{
	// calculate density for all particles except frozen ones according to E.q(6) from paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
	UnifiedParticle &p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		std::vector<int> &neighbor_indices = neighbor_indices_[i];

		// add own contribution
		p.density_ = kernel_self_ * fc_->initialMass;

		// iterate over all neighbors
		for (int j = 0; j < neighbor_indices.size(); ++j) {
			UnifiedParticle& neigh = particles_[neighbor_indices[j] ];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				// real neighbors, in this case we differentiate neighbor liquid particles from neighbor boundary particles 

				float kernelValue;
				if (neighbor_indices.size() < 1)	// TODO: should we use this trick? if so, what number should we use? 
				{
					// for particles with less neighbors, we use corrected density estimation
					kernelValue = my_kernel_->kernelM4Lut(dist) * neigh.shepard_filter_den_kernel_factor_;
				}
				else
				{
					// for the other ones, we use the normal summation
					kernelValue = my_kernel_->kernelM4Lut(dist);
				}

				if (neigh.type_ == LIQUID_PARTICLE)
				{
					// sum up contribution from liquid neighbors
					p.density_ += kernelValue * fc_->initialMass;
				}
				else if (neigh.type_ == FROZEN_PARTICLE)
				{
					// update weighted_volume for subsequent usage in boundary pressure force calculation

					// sum up contribution from boundary neighbors
					p.density_ += kernelValue * fc_->fluidRestDensity * neigh.weighted_volume_;

				}			
			}
		}

		// stats / debugging
		if(p.type_ == LIQUID_PARTICLE && p.density_ < min_dens_) min_dens_ = p.density_;
		if(p.type_ == LIQUID_PARTICLE && p.density_ > max_dens_) max_dens_ = p.density_;	
		dens_avg_timestep_ += p.density_;

		// correct for density near boundary, seems not good enough ***
		/*
		dist = p.position.y - collisionBox.getMin().y;
		if (dist < fc->globalSupportRadius) {
		dist = 0.5 * fc->globalSupportRadius - dist;
		float volume = 4.0/3.0 * M_PI * fc->globalSupportRadius*fc->globalSupportRadius*fc->globalSupportRadius;
		float cap = M_PI/6.0 * (3.0 * (fc->globalSupportRadius*fc->globalSupportRadius - (fc->globalSupportRadius - dist) * (fc->globalSupportRadius - dist)) + dist*dist) * dist;
		float factor = volume / (volume -cap);
		p.density *= factor;
		}
		*/

		// after having accumulated all contributions we can compute the pressure
		if(fc_->physicsType == 'o')
		{
			p.pressure_ = CalculatePressureSPH(p.density_);
		}
		else if(fc_->physicsType == 'w')
		{
			p.pressure_ = CalculatePressureWCSPH(p.density_);
		}
	}
}

void UnifiedPhysics::PredictionCorrectionStepVersatileCoupling()
{
	int i;
	int chunk = 100;
	int particlesSize = particles_.size();

	density_error_too_large_ = true; // loop has to be executed at least once

	int iteration = 0;
	while( (iteration < fc_->minLoops) || ((density_error_too_large_) && (iteration < fc_->maxLoops)) )	
	{
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)
			PredictPositionAndVelocity(i);

		max_predicted_density_ = 0.0; // loop termination criterion

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)
			ComputePredictedDensityAndPressure(i);

		// check loop termination criterion
		float densityErrorInPercent = MAX(0.1f * max_predicted_density_ - 100.0f, 0.0f); // 100/1000 * max_predicted_density_ - 100; 	

		if(fc_->printDebuggingInfo==1)
			std::cout << "ERROR: " << densityErrorInPercent << "%" << std::endl;

		// set flag to terminate loop if density error is smaller than requested
		if(densityErrorInPercent < fc_->maxDensityErrorAllowed) 
			density_error_too_large_ = false; // stop loop

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)			
			ComputeCorrectivePressureForceVersatilCoupling(i);

		iteration++;
	}

	// compute and print average and max number of iterations over whole simulation run
	int outCnt;
	if(fc_->printDebuggingInfo==1) outCnt = 1;
	else outCnt = 50;


	/*
	static int iterationPcisphStarted = iterationCounter-1; // needed to compute avgLoops while changing method during simulation
	avg_loops_sim_ += iteration;
	max_loops_sim_ = max(max_loops_sim_, iteration);
	if(iterationCounter % outCnt == 0)
	{
		cout << "nLoops done = " << iteration << " max_loops_sim_ = " << max_loops_sim_ << " avgLoops = " << avg_loops_sim_ / (iterationCounter-iterationPcisphStarted) << endl;
	}
	if(iteration > fc->maxLoops) cout << "maxLoops reached" << endl;
	//*/
}

void UnifiedPhysics::CalculateWeightedVolume(const int i)
{
	UnifiedParticle &p = particles_[i];
	std::vector<int> &neighbor_indices = neighbor_indices_[i];
	
	// calculate corrected kernels "Shepard Filter" according to E.q(2) from paper "Direct forcing for Lagrangian rigid-fluid coupling" for every particle
	float inverse_volume = 0.0f;
	// iterate over all particle neighbors and accumulate it's inverse volume & sum_Wc if any
	for (int j = 0; j < neighbor_indices.size(); ++j) 
	{
		UnifiedParticle& neigh = particles_[neighbor_indices[j] ];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius)
		{
			// all real neighbors including both liquid & frozen ones
			float kernelValue = my_kernel_->kernelM4Lut(dist);

			// calculate weighted volume of static boundary particles & rigid particles according to E.q(4) from paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
			if ( (p.type_ == FROZEN_PARTICLE || p.type_ == RIGID_PARTICLE) && (neigh.type_ == FROZEN_PARTICLE || neigh.type_ == RIGID_PARTICLE) )
			{	
				// for real boundary neighbor particles
				inverse_volume += kernelValue;
			}
		}
	}

	// update weighted boundary particle volume
	if (p.type_ == FROZEN_PARTICLE || p.type_ == RIGID_PARTICLE)
	{
		float V_bi = 0.0f;	// boundary particle volume
		if (inverse_volume != 0)
		{
			V_bi = 1.0 / inverse_volume;
		}
		else
		{
			std::cerr << "inverse_volume == 0!!! ERROR" << std::endl;
			abort();
		}
		// update weighted_volume for subsequent usage in boundary pressure force calculation
		// TODO: parallelize this
		p.weighted_volume_ = V_bi;
	}
}

void UnifiedPhysics::CalculateCorrectedDensitiesPressureVersatileCoupling(const int i)
{
	// calculate density for all particles except frozen ones according to E.q(6) from paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
	UnifiedParticle &p = particles_[i];
	std::vector<int> &neighbor_indices = neighbor_indices_[i];

	// add own contribution
	p.density_ = kernel_self_ * fc_->initialMass;

	// iterate over all neighbors
	for (int j = 0; j < neighbor_indices.size(); ++j) {
		UnifiedParticle& neigh = particles_[neighbor_indices[j] ];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius) {
			// real neighbors, in this case we differentiate neighbor liquid particles from neighbor boundary particles 

			float kernelValue = my_kernel_->kernelM4Lut(dist);

			if (neigh.type_ == LIQUID_PARTICLE)
			{
				// sum up contribution from liquid neighbors
				p.density_ += kernelValue * fc_->initialMass;
			}
			else if (neigh.type_ == FROZEN_PARTICLE)
			{
				// update weighted_volume for subsequent usage in boundary pressure force calculation

				// sum up contribution from boundary neighbors
				p.density_ += kernelValue * fc_->fluidRestDensity * neigh.weighted_volume_;
			}			
		}
	}

	// stats / debugging
	if(p.type_ == LIQUID_PARTICLE && p.density_ < min_dens_) min_dens_ = p.density_;
	if(p.type_ == LIQUID_PARTICLE && p.density_ > max_dens_) max_dens_ = p.density_;	
	dens_avg_timestep_ += p.density_;

	// correct for density near boundary, seems not good enough ***
	/*
	dist = p.position.y - collisionBox.getMin().y;
	if (dist < fc->globalSupportRadius) {
	dist = 0.5 * fc->globalSupportRadius - dist;
	float volume = 4.0/3.0 * M_PI * fc->globalSupportRadius*fc->globalSupportRadius*fc->globalSupportRadius;
	float cap = M_PI/6.0 * (3.0 * (fc->globalSupportRadius*fc->globalSupportRadius - (fc->globalSupportRadius - dist) * (fc->globalSupportRadius - dist)) + dist*dist) * dist;
	float factor = volume / (volume -cap);
	p.density *= factor;
	}
	*/

	// after having accumulated all contributions we can compute the pressure
	if(fc_->physicsType == 'o')
	{
		p.pressure_ = CalculatePressureSPH(p.density_);
	}
	else if(fc_->physicsType == 'w')
	{
		p.pressure_ = CalculatePressureWCSPH(p.density_);
	}
}

void UnifiedPhysics::CalculateForcesPerParticleStep1(const int i)
{
	// this method is called when using standard SPH

	/*
		(1) For each liquid particle, we add forces from liquid(pressure force & viscosity force & cohesion force), 
			from rigid-particles(versatile method), and from boundaries(simple penalty-based force/ Wall Weight 
			boundary pressure force/ versatile boundary particle pressure force & friction force)
		(2) For each rigid particle, we add forces from liquid & rigid particles & boundary force
	*/
	UnifiedParticle &p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE) // only calculate forces for non frozen particles
	{
		p.force_ = CalculateLiquidParticleForces(p, i);
		// add gravity force
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 
	}
	else if (p.type_ == RIGID_PARTICLE)
	{
		p.force_ = CalculateRigidParticleForces(p, i);
		// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
		// gravitational force to CalculateRigidBodyForceTorqueMomenta function
	}

	// TODO: we may add other controlling forces in here
}

void UnifiedPhysics::CalculateParticleForceVersatilCoupling(const int i)
{
	UnifiedParticle& p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		// We will be adding inside forces from liquid particles & outside forces exerted by boundary particles
		std::vector<int> &indices = neighbor_indices_[i];
		const float p_density = p.density_;
		const float invDensity = 1.0 / p_density;
		const float pVol = fc_->initialMass * invDensity;
		p.force_ = (0.0, 0.0, 0.0); 

		vmml::Vector3f force;
		force.set(0.0f, 0.0f, 0.0f);
		// iterate over all potential neighbors & add corresponding forces
		for (int j = 0; j < indices.size(); j++) 
		{
			UnifiedParticle &neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) 
			{
				// real neighbors
				const vmml::Vector3f x_ij = p.position_ - neigh.position_;
				const vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
				const vmml::Vector3f kernelGradient = x_ij * my_kernel_->kernelPressureGradLut(dist); // symmetric
				const vmml::Vector3f kernelViscosity = x_ij * my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
				if (neigh.type_ == LIQUID_PARTICLE)
				{	
					// ---------------------------versatile methods: Problem exists-----------------------------------------------
					//* versatile methods
					// sum up pressure force from liquid particles according to E.q(8) 
					// negative symmetry			
					float grad = pVol * pVol * p.pressure_;
					vmml::Vector3f tempPressureForce = kernelGradient * grad;
					force -= tempPressureForce;

					//* sum up viscosity force from liquid particles according to E.q(11)
					// negative symmetry
					float numerator = MAX(0.0, x_ij.dot(v_ij)); 
					float denominator = x_ij.lengthSquared() + fc_->epsilon_h_square;
					float pi = -1.0f * fc_->nu_liquid * numerator / denominator;
					force -= kernelViscosity * fc_->initialMass * fc_->initialMass * pi;
					//*/
				}
				else if (neigh.type_ == FROZEN_PARTICLE)
				{
					// ---------------------------versatile methods: Problem exists-----------------------------------------------
					/* versatile methods
					// sum up pressure force from boundary particles according to E.q(9) 
					float grad = fc->initialMass * fc->fluidRestDensity * neigh.weighted_volume * p.pressure / (p_density * p_density);
					vmml::Vector3f tempPressureForce = kernelGradient * grad;
					force -= tempPressureForce;

					// sum up viscosity force from boundary particles according to E.q(13)
					float numerator = MAX(0.0, x_ij.dot(v_ij)); 
					float denominator = x_ij.lengthSquared() + fc->epsilon_h_square;
					float pi = -1.0f * fc->nu_rigid_liquid * numerator / denominator;
					force -= kernelViscosity * fc->initialMass * fc->fluidRestDensity * neigh.weighted_volume * pi;
					//*/
				}	

				// calculate temperature diffusion change per time	
				/*
				float dTemp = neigh.temperature - p.temperature;
				dTemp *= kernelVisc;
				p.deltaTemperaturePerTime += dTemp * nVol;
				*/
			}
		}// end over all neighbors

		// add force for each particle
		p.force_ = force;

		// add gravity
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 
	}
}

void UnifiedPhysics::CalculateBoundaryForceMonaghan(const int i)
{
	UnifiedParticle& p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		// We will be adding inside forces from liquid particles & outside forces exerted by boundary particles
		std::vector<int> &indices = neighbor_indices_[i];
		const float p_density = p.density_;
		const float invDensity = 1.0 / p_density;
		const float pVol = fc_->initialMass * invDensity;
		p.force_ = (0.0, 0.0, 0.0); 

		vmml::Vector3f force, pressureForce;
		force.set(0.0f, 0.0f, 0.0f);
		pressureForce.set(0.0f, 0.0f, 0.0f);
		// iterate over all potential neighbors & add corresponding forces
		for (int j = 0; j < indices.size(); j++) 
		{
			UnifiedParticle &neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) 
			{
				// real neighbors
				float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
				float nVol = fc_->initialMass / neigh.density_;
				const vmml::Vector3f distVec = p.position_ - neigh.position_;

				if (neigh.type_ == LIQUID_PARTICLE)
				{	
					//*
					vmml::Vector3f kernelGradient = distVec * my_kernel_->kernelPressureGradLut(dist); // symmetric
					// sum up pressure force according to Monaghan
					// negative symmetry
					float grad = p.pressure_ * invDensity * invDensity + neigh.pressure_ / (neigh.density_*neigh.density_);
					vmml::Vector3f tempPressureForce = kernelGradient * grad * fc_->initialMass * fc_->initialMass;
					pressureForce -= tempPressureForce;
					force -= tempPressureForce;
					//*/

					//*
					// compute artificial viscosity according to MCG03
					// negative symmetry
					vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
					//force -= v_ij * pVol * nVol * fc->fluidViscConst * kernelVisc;
					force -= v_ij * pVol * nVol * fc_->fluidViscConst * kernelVisc;
					//*/
				}
				else if (neigh.type_ == FROZEN_PARTICLE)
				{
					const vmml::Vector3f n_k = neigh.normalized_normal_;
					float normalDistance = distVec.dot(n_k);
					float tangentialDistance = sqrt(distVec.lengthSquared() - normalDistance * normalDistance); 
					float B_x = BoundaryForceBxy(tangentialDistance, normalDistance, fc_->globalSupportRadius, fc_->particleSpacing);
					force += n_k * 0.5f * B_x; // E.q(10.5) from Monaghan 2005 "Smoothed particle hydrodynamics" P1749
				}	

				// calculate temperature diffusion change per time	
				/*
				float dTemp = neigh.temperature - p.temperature;
				dTemp *= kernelVisc;
				p.deltaTemperaturePerTime += dTemp * nVol;
				*/
			}
		}// end over all neighbors

		// calculate force for each particle
		p.force_ = force;

		//注意：PCISPHCudaSharedMemory中使用的是force, fluid-solid code 中使用的是 force density 注意timeIntegration中由这个区别所引起的问题
		// add gravity
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 
		//p.force[1] -= p.density * fc->gravityConst; 

		p.pressure_force_ = pressureForce;
	}
}

void UnifiedPhysics::CalculateForcesPerParticleStep2(const int i)
{
	// Note: Right now, this step can't be parallelized TODO: change the algorithm to make it parallelizable
	// this method is called when using standard SPH
	// finish calculation in calculateForcesStep 1, sum individual forces

	// calculate surface tension
	// f^surface = - sigma * lap c_s * n / |n| where
	// c_s is the smoothed colour field  
	// n = sum_{all_neighbors} m_j / rho_j * grad W(r-r_j,h)

	UnifiedParticle &p = particles_[i];
	RigidBody* parentBody = p.parent_rigidbody_;

	// multiply with temperature diffusion constant
	//p.deltaTemperaturePerTime *= fc->diffusionTempConst;
}

vmml::Vector3f UnifiedPhysics::CalculateLiquidParticleForces(UnifiedParticle& p, const int i)
{
	std::vector<int> &indices = neighbor_indices_[i];

	const float invDensity = 1.0 / p.density_;
	const float pVol = fc_->initialMass * invDensity;
	p.force_ = (0.0, 0.0, 0.0); 

	vmml::Vector3f force, pressureForce;
	force.set(0.0f, 0.0f, 0.0f);
	pressureForce.set(0.0f, 0.0f, 0.0f);

	// iterate over all potential neighbors
	for (int j = 0; j < indices.size(); j++) 
	{
		UnifiedParticle &neigh = particles_[indices[j]];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius) 
		{
			// real neighbors
			float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
			float nVol = fc_->initialMass / neigh.density_;
			const vmml::Vector3f distVec = p.position_ - neigh.position_;

			// we only calculate pressure & viscous force from inner liquid
			if (neigh.type_ == LIQUID_PARTICLE)
			{
				//*
				vmml::Vector3f kernelGradient = distVec * my_kernel_->kernelPressureGradLut(dist); // symmetric
				// sum up pressure force according to Monaghan
				// negative symmetry
				float grad = p.pressure_ * invDensity * invDensity + neigh.pressure_ / (neigh.density_*neigh.density_);
				vmml::Vector3f tempPressureForce = kernelGradient * grad * fc_->initialMass * fc_->initialMass;
				pressureForce -= tempPressureForce;
				force -= tempPressureForce;
				//*/

				//*
				// compute artificial viscosity according to MCG03
				// negative symmetry
				vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
				//force -= v_ij * pVol * nVol * fc->fluidViscConst * kernelVisc;
				force -= v_ij * pVol * nVol * fc_->fluidViscConst * kernelVisc;
				//*/

				//*
				if (fc_->useSurfaceTension && dist >= fc_->globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
				{
					float splineCoefficient = my_kernel_->kernelSplineSurfaceTension(dist, fc_->globalSupportRadius);

					force -= CalculateSurfaceTensionForcePCISPH(dist, p.position_, neigh.position_, splineCoefficient);
				}
				//*/
			}

			// calculate temperature diffusion change per time	
			/*
			float dTemp = neigh.temperature - p.temperature;
			dTemp *= kernelVisc;
			p.deltaTemperaturePerTime += dTemp * nVol;
			*/
		}
	}// end over all neighbors

	p.pressure_force_ = pressureForce;

	//*
	// add boundary forces 
	if (fc_->addBoundaryForce)
		force += BoxBoundaryForce(i);
	//p.force += GetBoundaryPressureForceWallWeight(i);	// TODO: this version is incorrect!!!
	//*/

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateRigidParticleForces(UnifiedParticle& p, const int i)
{
	std::vector<int> &indices = neighbor_indices_[i];

	const float invDensity = 1.0 / p.density_;
	const float pVol = fc_->initialMass * invDensity;

	vmml::Vector3f force;
	force.set(0.0f, 0.0f, 0.0f);

	RigidBody* parentBody = p.parent_rigidbody_;

	// iterate over all potential neighbors
	for (int j = 0; j < indices.size(); j++) 
	{
		UnifiedParticle &neigh = particles_[indices[j]];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius) 
		{
			// real neighbors
			float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
			float nVol = fc_->initialMass / neigh.density_;
			// don't calculate any forces if both particles belong to the same rigid body, and the body has an inverted inertia
			// also don't calculate any forces between liquid particles & frozen particles
			if(neigh.type_ == LIQUID_PARTICLE)
			{
				// TODO: add force from liquid particle using versatile method
			}
			else if (neigh.type_ == RIGID_PARTICLE)
			{
				// add force from rigid particle
				// only calculate forces between rigid particles if they belong to different rigid body
				const float overlap = 2.0f * fc_->particleRadius - dist;
				if ( (p.parent_rigidbody_ != neigh.parent_rigidbody_) && overlap > 0.0f )
				{
					// f = f_spring + f_damping + f_shear "Real-Time Rigid Body Simulation on GPUs E.q(10) & E.q(11) & E.q(12)
					// f_spring = -k * (2*particleRadius - dist) * r_ij/|r_ij| Here we have to make sure f_spring is pointing to p
					// f_damping = eta * (v_i - v_j)
					// f_shear = k_t * v_ij_tangential
					const vmml::Vector3f r_ij = neigh.position_ - p.position_;									// relative position of particle j with respect to particle i
					const vmml::Vector3f r_ij_normalize = r_ij.getNormalized();
					const vmml::Vector3f f_spring = -r_ij_normalize * fc_->springCoefficient * overlap;

					const vmml::Vector3f v_ij = neigh.velocity_ - p.velocity_;							// relative velocity of particle j with respect to particle i
					const vmml::Vector3f f_damping = v_ij * fc_->dampingCoefficient;

					const vmml::Vector3f v_ij_tangential = v_ij - r_ij_normalize * (v_ij.dot(r_ij_normalize) );
					const vmml::Vector3f f_shear = v_ij_tangential * fc_->shearCoefficient;

					force += f_spring + f_damping + f_shear;
				}
			} 

			// calculate temperature diffusion change per time	
			/*
			float dTemp = neigh.temperature - p.temperature;
			dTemp *= kernelVisc;
			p.deltaTemperaturePerTime += dTemp * nVol;
			*/
		}
	}// end over all neighbors

	// Add force from boundaries (container) 		
	force += CalculateRigidParticleBoundaryForces(p.position_, p.velocity_);

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateRigidParticleBoundaryForces(const vmml::Vector3f& pos, const vmml::Vector3f& vel)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);
	const vmml::Vector3f& minCollBox = fc_->collisionBox.getMin();
	const vmml::Vector3f& maxCollBox = fc_->collisionBox.getMax();
	float overlap = 0.0f;

	bool collisionOccured = false;

	// ground collision
	overlap = minCollBox.y - (pos.y - fc_->particleRadius);
	if ( overlap > 0.0f )
	{
		collisionOccured = true;
		force.y += fc_->springCoefficientBoundary * overlap;
	}
	
	// ceiling collision
	overlap = pos.y + fc_->particleRadius - maxCollBox.y;
	if (overlap > 0.0f)
	{
		collisionOccured = true;
		force.y -= fc_->springCoefficientBoundary * overlap ;
	}
	
	// left wall collision
	overlap = minCollBox.x - (pos.x - fc_->particleRadius);
	if (overlap > 0.0f)
	{
		collisionOccured = true;
		force.x += fc_->springCoefficientBoundary * overlap;
	}
	
	// right wall collision
	overlap = pos.x + fc_->particleRadius - maxCollBox.x;
	if (overlap > 0.0f)
	{
		collisionOccured = true;
		force.x -= fc_->springCoefficientBoundary * overlap;
	}

	// back wall collision
	overlap = minCollBox.z - (pos.z - fc_->particleRadius);
	if (overlap > 0.0f)
	{
		collisionOccured = true;
		force.z += fc_->springCoefficientBoundary * overlap;
	}
	
	// front wall collision
	overlap = pos.z + fc_->particleRadius - maxCollBox.z;
	if (overlap > 0.0f)
	{
		collisionOccured = true;
		force.z -= fc_->springCoefficientBoundary * overlap;
	}

	// Damping
	if (collisionOccured)
	{
		force -= vel * fc_->dampingCoefficient;
	}
	
	return force;
}

void UnifiedPhysics::CalculateForcesWithContactForces(const int i)
{
	// this method is called when using standard SPH
	UnifiedParticle &p = particles_[i];
	if (p.type_ != FROZEN_PARTICLE) // only calculate forces for non frozen particles
	{
		std::vector<int> &indices = neighbor_indices_[i];

		const float invDensity = 1.0 / p.density_;
		const float pVol = fc_->initialMass * invDensity;
		p.force_ = (0.0, 0.0, 0.0); 

		vmml::Vector3f force, pressureForce;
		force.set(0.0f, 0.0f, 0.0f);
		pressureForce.set(0.0f, 0.0f, 0.0f);

		RigidBody* parentBody = p.parent_rigidbody_;

		// iterate over all potential neighbors
		for (int j = 0; j < indices.size(); j++) 
		{
			UnifiedParticle &neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) 
			{
				// real neighbors
				float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
				float nVol = fc_->initialMass / neigh.density_;
				// don't calculate any forces if both particles belong to the same rigid body, and the body has an inverted inertia
				if(!parentBody || parentBody!=neigh.parent_rigidbody_)
				{
					const vmml::Vector3f distVec = p.position_ - neigh.position_;

					// add particle-particle "contact" force  F_p = F_n + F_t  p <-> x_1  &  neigh <-> x_2 from 
					// paper "Particle-Based Simulation of granular Materials" when they are in contact
					// add additional forces between rigid particles from different rigid bodies (both dynamic rigid bodies or static/frozen particles)
					if ( dist < fc_->collisionDistThreshold && ((parentBody && neigh.parent_rigidbody_) || neigh.type_ == FROZEN_PARTICLE) )
					{
						// calculate normal force F_n
						// formula(1)
						float ksi = std::max(fc_->collisionDistThreshold - dist, 0.0f); 

						// formula(2)
						// distVec = p->position - neigh->position;
						vmml::Vector3f X2X1 = distVec * (-1.0f);
						X2X1.normalize();
						vmml::Vector3f N = X2X1;

						//formula(3)~(5)
						// Note: V_1 and V_2 are the velocities of each particle at the contact point, not at the center of mass
						vmml::Vector3f V_1 = p.velocity_; // approximation
						vmml::Vector3f V_2 = neigh.velocity_; // approximation
						vmml::Vector3f V = V_1 - V_2;
						float ksiDot = V.dot(N);
						vmml::Vector3f V_t = V - N * ksiDot;

						// calculate F_n = f_n * N
						// formula (8) linear model
						float f_n = -fc_->k_d * ksiDot - fc_->k_r * ksi;
						vmml::Vector3f F_n = N * f_n;

						// formula (18) calculate shear force F_t
						float temp = std::min(fc_->mu * f_n, fc_->k_t * V_t.normalize());
						vmml::Vector3f F_t = V_t * (-temp); // V_t has aleady normalized through V_t.normalize() in previous line

						// contact force F = F_n + F_t
						vmml::Vector3f F_contact = F_n + F_t;

						force += F_t;
					}
					else
					{
						//*
						vmml::Vector3f kernelGradient = distVec * my_kernel_->kernelPressureGradLut(dist); // symmetric
						// sum up pressure force according to Monaghan
						// negative symmetry
						float grad = p.pressure_ * invDensity * invDensity + neigh.pressure_ / (neigh.density_*neigh.density_);
						vmml::Vector3f tempPressureForce = kernelGradient * grad * fc_->initialMass * fc_->initialMass;
						pressureForce -= tempPressureForce;
						force -= tempPressureForce;
						//*/

						//*
						// compute artificial viscosity according to MCG03
						// negative symmetry
						vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
						//force -= v_ij * pVol * nVol * fc->fluidViscConst * kernelVisc;
						force -= v_ij * pVol * nVol * fc_->fluidViscConst * kernelVisc;
						//*/
					
					}

				}

				// calculate temperature diffusion change per time	
				/*
				float dTemp = neigh.temperature - p.temperature;
				dTemp *= kernelVisc;
				p.deltaTemperaturePerTime += dTemp * nVol;
				*/
			}
		}// end over all neighbors

		// calculate force for each particle
		p.force_ = force;

		//注意：PCISPHCudaSharedMemory中使用的是force, fluid-solid code 中使用的是 force density 注意timeIntegration中由这个区别所引起的问题
		// add gravity
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 
		//p.force[1] -= p.density * fc->gravityConst; 

		p.pressure_force_ = pressureForce;

		//*
		// add boundary forces 
		if (fc_->addBoundaryForce)
			p.force_ += BoxBoundaryForce(i);
		//*/
	}
}

void UnifiedPhysics::CalculateForcesWithoutBoundaryControllingForces(const int i)
{
	// calculate all forces except boundary controlling forces
	// this method is called when using standard SPH
	UnifiedParticle &p = particles_[i];
	if (p.type_ != FROZEN_PARTICLE) // only calculate forces for non frozen particles
	{
		std::vector<int> &indices = neighbor_indices_[i];

		const float invDensity = 1.0 / p.density_;
		const float pVol = fc_->initialMass * invDensity;
		p.force_ = (0.0, 0.0, 0.0); 

		vmml::Vector3f force, pressureForce;
		force.set(0.0f, 0.0f, 0.0f);
		pressureForce.set(0.0f, 0.0f, 0.0f);

		RigidBody* parentBody = p.parent_rigidbody_;

		// iterate over all potential neighbors
		for (int j = 0; j < indices.size(); j++) 
		{
			UnifiedParticle &neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) 
			{
				// real neighbors
				float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
				float nVol = fc_->initialMass / neigh.density_;

				// for each non rigid particle or rigid particles from different rigid bodies
				// don't calculate any forces if both particles belong to the same rigid body, and the body has an inverted inertia
				if(!parentBody || parentBody!=neigh.parent_rigidbody_)
				{
					const vmml::Vector3f distVec = p.position_ - neigh.position_;

					//*
					vmml::Vector3f kernelGradient = distVec * my_kernel_->kernelPressureGradLut(dist); // symmetric
					// sum up pressure force according to Monaghan
					// negative symmetry
					float grad = p.pressure_ * invDensity * invDensity + neigh.pressure_ / (neigh.density_*neigh.density_);
					vmml::Vector3f tempPressureForce = kernelGradient * grad * fc_->initialMass * fc_->initialMass;
					pressureForce -= tempPressureForce;
					force -= tempPressureForce;
					//*/

					//*
					// compute artificial viscosity according to MCG03
					// negative symmetry
					vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
					//force -= v_ij * pVol * nVol * fc->fluidViscConst * kernelVisc;
					force -= v_ij * pVol * nVol * fc_->fluidViscConst * kernelVisc;
					//*/

				}

				// calculate temperature diffusion change per time	
				/*
				float dTemp = neigh.temperature - p.temperature;
				dTemp *= kernelVisc;
				p.deltaTemperaturePerTime += dTemp * nVol;
				*/
			}
		}// end over all neighbors

		// calculate force for each particle
		p.force_ = force;

		//注意：PCISPHCudaSharedMemory中使用的是force, fluid-solid code 中使用的是 force density 注意timeIntegration中由这个区别所引起的问题
		// add gravity
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 
		//p.force[1] -= p.density * fc->gravityConst; 

		p.pressure_force_ = pressureForce;

	}
}

void UnifiedPhysics::CalculateForcesPerParticleStep3ForElasticBodies(const int i)
{
	// this method is called when using standard SPH
	// calculate forces for elastic bodies

	UnifiedParticle &p = particles_[i];
	if (p.type_ != FROZEN_PARTICLE && !solid_neighs_.empty() && !solid_neighs_[i].empty()) // only calculate forces for non frozen particles which has at least one solid neighbor
	{
		// gradient of displacement = [gradX, gradY, gradZ]
		// gradX = gradient of (displacement in the x dimension)
		vmml::Vector3f gradX;
		gradX.set(0.0f, 0.0f, 0.0f);
		vmml::Vector3f gradY;
		gradY.set(0.0f, 0.0f, 0.0f);
		vmml::Vector3f gradZ;
		gradZ.set(0.0f, 0.0f, 0.0f);

		RigidBody* parentBody = p.parent_rigidbody_;

		// if p and all of its solidNeighs are part of the same rigid object, elasticity is not necessary,
		// if it belongs to a rigid body, all solidNeighs that have a rigid body are parts of the same one.
		bool elasticityNeeded = (parentBody==NULL);

		std::vector<int>::iterator endIter = solid_neighs_[i].end();
		std::vector<vmml::Vector3f>::iterator distIter = solid_neigh_distances_[i].begin();
		for (std::vector<int>::iterator n = solid_neighs_[i].begin(); n != endIter; ++n, ++distIter)
		{
			// all neighbors are used for calculating grad Displacement, even if p & neigh belong to the same rigid body

			UnifiedParticle* neigh = &particles_[*n];
			vmml::Vector3f distVec = *distIter;

			float neighVol = neigh->solid_volume_;
			vmml::Vector3f deltaDisplacement = vmml::Vector3f(neigh->position_ - p.position_) + distVec; // displacement_j - displacement_i  "A Unified Particle Model for Fluid-Solid Interactions"Eq.(15)

			elasticityNeeded |= (neigh->parent_rigidbody_==NULL);
			/*/ MLS 
			// when using this, also uncomment the CalculateInvertedMomentMatrix-block in the constructor; the SPH block below should be commented out.
			// The MLS implementation does not support melting or freezing.
			distVec *= -1; // this formula uses x_ij = x_j - x_i
			float sqLength = distVec.lengthSquared();
			vmml::Vector3f weightedDistVec = distVec*KernelDensity(sqLength);
			weightSum=1;
			vmml::Vector3f gradXMls = weightedDistVec * deltaDisplacement[0];
			vmml::Vector3f gradYMls = weightedDistVec * deltaDisplacement[1];
			vmml::Vector3f gradZMls = weightedDistVec * deltaDisplacement[2];
			neigh->strainFactor.set(0.0f, 0.0f, 0.0f);
			for(int i=0;i<3;++i)
			{
			for(int j=0;j<3;++j)
			{
			neigh->strainFactor[i] += p.invertedMomentMatrix[i][j] * weightedDistVec[j];
			gradX[i] += p.invertedMomentMatrix[i][j]*gradXMls[j];
			gradY[i] += p.invertedMomentMatrix[i][j]*gradYMls[j];
			gradZ[i] += p.invertedMomentMatrix[i][j]*gradZMls[j];
			}
			}
			//*/
			// SPH computation of the gradient of displacement
			float distLength = distVec.length();
			neigh->strain_factor_ = distVec*neighVol*KernelElasticGrad(distLength);
			gradX += neigh->strain_factor_*deltaDisplacement[0];
			gradY += neigh->strain_factor_*deltaDisplacement[1];
			gradZ += neigh->strain_factor_*deltaDisplacement[2];
			//*/
		}
		if(!elasticityNeeded) return; // stop calculating elastic forces if it's not necessary

		// d=displacement
		// strain = 1/2*(grad d + (grad d [transpose]) + grad d x (grad d [transpose]))
		// NOTE: factor 1/2 has been introduced later because it does not appear in [MKN*04], but
		// in other papers -> should other parts of the formula be adapted, too?
		// strain is symmetric, so a 1x6 vector can be used instead of a 3x3 matrix
		// strain={strainXX,strainYY,strainZZ,strainXY,strainYZ,strainZX}
		// elasticStrain=strain-plasticStrain
		float elasticStrain[6] ={ gradX[0]+(gradX[0]*gradX[0]+gradY[0]*gradY[0]+gradZ[0]*gradZ[0])*0.5f - p.plastic_strain_[0],
			gradY[1]+(gradX[1]*gradX[1]+gradY[1]*gradY[1]+gradZ[1]*gradZ[1])*0.5f - p.plastic_strain_[1],
			gradZ[2]+(gradX[2]*gradX[2]+gradY[2]*gradY[2]+gradZ[2]*gradZ[2])*0.5f - p.plastic_strain_[2],
			(gradX[1]+gradY[0]+gradX[0]*gradX[1]+gradY[0]*gradY[1]+gradZ[0]*gradZ[1])*0.5f - p.plastic_strain_[3],
			(gradY[2]+gradZ[1]+gradX[1]*gradX[2]+gradY[1]*gradY[2]+gradZ[1]*gradZ[2])*0.5f - p.plastic_strain_[4],
			(gradZ[0]+gradX[2]+gradX[2]*gradX[0]+gradY[2]*gradY[0]+gradZ[2]*gradZ[0])*0.5f - p.plastic_strain_[5],
		};

		UpdateStrain(&p, elasticStrain);

		// stress = hookeMatrix x elasticStrain
		float stress[6];
		for(int i=0; i<6; ++i)
		{
			stress[i]=0;
			for(int j=0; j<6; ++j)
			{
				stress[i] += p.hooke_matrix_[i][j]*elasticStrain[j];
			}
		}

		vmml::Vector3f jacobi[3] = {gradX, gradY, gradZ};
		jacobi[0][0]++;
		jacobi[1][1]++;
		jacobi[2][2]++;

		// forceMatrix = 2*vol_i*Jacobi_i*stress_i
		float stressMatrix[3][3] = { stress[0], stress[3], stress[5],
			stress[3], stress[1], stress[4],
			stress[5], stress[4], stress[2]			
		};
		float forceMatrix[3][3];
		memset(forceMatrix, 0, 9*sizeof(float));
		float volume = p.solid_volume_;
		for(int i=0; i<3; ++i)
		{
			for(int j=0; j<3; ++j)
			{
				for(int k=0; k<3; ++k)
				{
					forceMatrix[i][j] += volume * jacobi[i][k] * stressMatrix[k][j]; // there would be a *2, but it's left out: f_j= (f_ij+f_ji)/2 = (f_ij/2+f_ji/2)
				}
			}
		}

		// volume conserving force
		vmml::Vector3f volConservingMatrix[3];
		if(p.volume_conserving_coeff_ > 0.0f)
		{
			float detJacobi = 0; // determinant of the jacobi matrix
			for(int i=0; i<3; ++i)
			{
				float summand = 1;
				float subtrahend = 1;
				int row = i;
				for(int j=0; j<3; ++j)
				{
					summand *= jacobi[row][j];
					subtrahend *= jacobi[2-row][j];
					row = (row+1)%3;
				}
				detJacobi += summand - subtrahend;
			}

			float conservingFactor = p.solid_volume_ * p.volume_conserving_coeff_ * (detJacobi-1)/2; // f_j= (f_ij+f_ji)/2 -> div by 2
			volConservingMatrix[0] = (jacobi[1].cross(jacobi[2]) ) * conservingFactor;
			volConservingMatrix[1] = (jacobi[2].cross(jacobi[0]) ) * conservingFactor;
			volConservingMatrix[2] = (jacobi[0].cross(jacobi[1]) ) * conservingFactor;
		}
		else
		{
			volConservingMatrix[0].set(0.0f, 0.0f, 0.0f);
			volConservingMatrix[1].set(0.0f, 0.0f, 0.0f);
			volConservingMatrix[2].set(0.0f, 0.0f, 0.0f);
		}
		vmml::Vector3f sumElasticForce;
		sumElasticForce.set(0.0f, 0.0f, 0.0f);
		for (std::vector<int>::iterator n = solid_neighs_[i].begin(); n != endIter; ++n)
		{
			
			UnifiedParticle* neigh = &particles_[*n];

			RigidBody* neighParentBody = neigh->parent_rigidbody_;
			// don't calculate any forces if both particles belong to the same rigid body
			if(parentBody && neighParentBody) continue;

			vmml::Vector3f strainFactor = neigh->strain_factor_;

			vmml::Vector3f elasticForce;
			elasticForce.set(0.0f, 0.0f, 0.0f);
			for(int i=0; i<3; ++i)
			{
				for(int j=0; j<3; ++j)
				{
					elasticForce[i] += (forceMatrix[i][j]+volConservingMatrix[i][j]) * strainFactor[j];
				}
			}

			// force acting on the body neighbours
			if(neighParentBody)
			{
				vmml::Vector3f negativeForce = elasticForce*-1;
				neighParentBody->ApplyForce(negativeForce);
				neighParentBody->ApplyTorque(negativeForce, *neigh);
			}
			else
			{
				vmml::Vector3f temp = elasticForce * (neigh->density_ / neigh->particle_mass_ );
				neigh->force_ -= vmml::Vector3f(temp[0], temp[1], temp[2]);
			}
			sumElasticForce += elasticForce;

		}
		// force acting on the particle itself
		if(parentBody)
		{
			parentBody->ApplyForce(sumElasticForce);
			parentBody->ApplyTorque(sumElasticForce, p);
		}
		else
		{
			vmml::Vector3f temp = sumElasticForce * (p.density_ / p.particle_mass_);
			p.force_ += vmml::Vector3f(temp[0], temp[1], temp[2]);
		}
	}
}

void UnifiedPhysics::CalculateRigidBodyForceTorqueMomenta(const int i)
{
	static const float deltaT = fc_->deltaT;

	// calculate linear momentum and angular momentum of each rigid object
	// Euler steps are used for now
	RigidBody* body = rigid_bodies_[i];
	if (body)
	{
		//* Step 1
		// Iterate over all particles and calculate total force & torque from particles
		std::vector<int> &particleIndicsArray = body->rigid_particle_indices();
		const int num_particles = particleIndicsArray.size(); 
		for (int j = 0; j < num_particles; ++j)
		{
			int index = particleIndicsArray[j];
			UnifiedParticle& p = particles_[index];
			
			// apply force & torque to parent rigid body for each particles
			body->ApplyForce(p.force_);
			body->ApplyTorque(p.force_, p);

			// add gravity force
			vmml::Vector3f gravityForce(0,-p.particle_mass_ * fc_->gravityConst,0);
			body->ApplyForce(gravityForce);
		}
		//*/

		if (fc_->useOldRigidBodyMethod)
		{
			///*-------------Old Method Step 2: update rigid body information --------------------------------------------
			body->set_center_of_mass(body->rigidbody_pos());
			body->set_rigidbody_pos(body->rigidbody_pos() + body->velocity() * deltaT);
			const vmml::Vector3f vel = body->velocity() + body->force() * (deltaT / body->mass()); // body has a force, not force density
			body->set_velocity(vel); 

			// compute Ri according to (Eq. 5.13) from "Simulation of Fluid-Solid Interaction" 
			vmml::Vector3f quatVec = body->angular_velocity()*(deltaT/2); // vector part of rotation quaternion
			float lengthFactor = 2/(quatVec.lengthSquared()+1); // 2 / squared length of quaternion

			Matrix3x3 tmp_rot(body->rotation_matrix());
			tmp_rot.elements[0][0] = 1-lengthFactor*(quatVec[1]*quatVec[1]+quatVec[2]*quatVec[2]);
			tmp_rot.elements[1][1] = 1-lengthFactor*(quatVec[0]*quatVec[0]+quatVec[2]*quatVec[2]);
			tmp_rot.elements[2][2] = 1-lengthFactor*(quatVec[1]*quatVec[1]+quatVec[0]*quatVec[0]);
			tmp_rot.elements[0][1] = lengthFactor*(quatVec[0]*quatVec[1]-quatVec[2]);
			tmp_rot.elements[1][0] = lengthFactor*(quatVec[0]*quatVec[1]+quatVec[2]);
			tmp_rot.elements[0][2] = lengthFactor*(quatVec[0]*quatVec[2]+quatVec[1]);
			tmp_rot.elements[2][0] = lengthFactor*(quatVec[0]*quatVec[2]-quatVec[1]);
			tmp_rot.elements[1][2] = lengthFactor*(quatVec[1]*quatVec[2]-quatVec[0]);
			tmp_rot.elements[2][1] = lengthFactor*(quatVec[1]*quatVec[2]+quatVec[0]);
			body->set_rotation_matrix(tmp_rot);

			body->set_angular_momentum(body->angular_momentum() + body->torque() * deltaT);
			body->set_inverted_inertia(body->rotation_matrix() * body->inverted_inertia() * body->rotation_matrix().GetTransposedMatrix());
			// rotate inertia as well, although it is not used, because this avoids recalculating it by iterating over all particles of the body when melting/freezing
			//body->inertia = body->rotationMatrix * body->inertia * body->rotationMatrix.GetTransposedMatrix();
			body->set_angular_velocity(body->inverted_inertia() * body->angular_momentum());

			body->set_force(vmml::Vector3f(0.0f, 0.0f, 0.0f));
			body->set_torque(vmml::Vector3f(0.0f, 0.0f, 0.0f));
			// --------------------------------------------------------------------------------------------------------------------------------*/
		} 
		else
		{
			// construct the rotation matrix using quaternion maths
			// TODO: use vmml::Vector3f quaternion to replace quatVec in here 
			//* Step 2: calculate momenta using force & torque
			body->set_center_of_mass(body->rigidbody_pos());
			const float terminalMomentum = body->mass() * fc_->terminalSpeed;

			body->UpdateMomenta(fc_->deltaT, terminalMomentum);

			// rigid body position & rotation update
			body->PerformStep(fc_->deltaT);
			//*/

			body->UpdateRotationMatrix();

			body->set_force(vmml::Vector3f(0.0f, 0.0f, 0.0f));
			body->set_torque(vmml::Vector3f(0.0f, 0.0f, 0.0f));

		}
	}
}

void UnifiedPhysics::UpdateLiquidParticlePosVelSphCPU(const int i)
{
	UnifiedParticle &p = particles_[i];
	if(p.type_ == LIQUID_PARTICLE)
	{	
		// in case of PCISPH add the pressureForce to total force
		if(fc_->physicsType == 'i') 
			p.force_ += p.correction_pressure_force_;

		static const float invMass = 1.0 / fc_->initialMass;
		static const float deltaT = fc_->deltaT;
		/************************************************************************/
		/*                    Symplectic Euler Integration                      */
		/************************************************************************/
		//*
		p.velocity_ += p.force_ * invMass * deltaT;
		p.position_ += p.velocity_ * deltaT;
		//*/

		/************************************************************************/
		/*                        Leapfrog integration                          */
		/************************************************************************/
		/*
		vmml::Vector3f vel_half_previous = p.velocity_leapfrog;								// read old velocity_leapfrog			 v(t-1/2)
		vmml::Vector3f vel_half_next = vel_half_previous + p.force * invMass * deltaT;		// calculate new velocity_leapfrog		 v(t+1/2) = v(t-1/2) + a(t) * dt	
		p.velocity_leapfrog = vel_half_next;											// update new velocity_leapfrog 
		p.velocity = (vel_half_previous + vel_half_next) * 0.5f;						// update new velocity 					 v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5
		p.position += vel_half_next * deltaT;											// update new position					 p(t+1) = p(t) + v(t+1/2) * dt	
		//*/
	}

	// diffusion: Euler step
	/*
	p.temperature += p.deltaTemperaturePerTime * deltaT;
	if(p.temperature > 100)
	p.temperature = 100;
	*/
}

void UnifiedPhysics::SynRigidParticlesPerRigidBody(RigidBody* body)
{
	if (fc_->useOldRigidBodyMethod)
	{
		//* Old Method
		const int num_rigid_particles = body->rigid_particle_indices().size();
		for (int j = 0; j < num_rigid_particles; ++j)
		{
			int p_index = body->rigid_particle_indices()[j];
			UnifiedParticle &p = particles_[p_index];	
			if(body)
			{
				vmml::Vector3f relativePosition = p.position_ - body->old_center_of_mass();
				vmml::Vector3f newRelativePosition = body->rotation_matrix() * relativePosition;
				// v = linear velocity + tangential velocity
				p.velocity_ = body->velocity() + (body->angular_velocity()).cross(newRelativePosition);
				p.position_ = body->rigidbody_pos() + newRelativePosition;
			}
		}
		//*/
	} 
	else
	{
		//*
		// apply rotation to rigid particles
		const int num_rigid_particles = body->rigid_particle_indices().size();
		for (int j = 0; j < num_rigid_particles; ++j)
		{
			int p_index = body->rigid_particle_indices()[j];
			UnifiedParticle &p = particles_[p_index];	
			if(body)
			{
				const vmml::Vector3f newRelativePosition = body->rotation_matrix() * p.init_relative_pos_;
				
				// v = linear velocity + tangential velocity
				p.velocity_ = body->velocity() + (body->angular_velocity()).cross(newRelativePosition);
				p.position_ = body->rigidbody_pos() + newRelativePosition;
			}
		}
		//*/
	}

	// diffusion: Euler step
	/*
	p.temperature += p.deltaTemperaturePerTime * deltaT;
	if(p.temperature > 100)
	p.temperature = 100;
	*/
}

vmml::Vector3f UnifiedPhysics::BoxBoundaryForce(const int i)
{
	static const float forceDistance = fc_->forceDistance;
	static const float invForceDist = 1.0 / fc_->forceDistance;
	static const float forceStrength = fc_->maxBoundaryForce;
	static const vmml::Vector3f minCB = fc_->collisionBox.getMin();
	static const vmml::Vector3f maxCB = fc_->collisionBox.getMax();
	float distToWall, factor;
	vmml::Vector3f force(0.0);
	UnifiedParticle &p = particles_[i];

	// ground-ceiling
	if (p.position_.y < minCB.y+forceDistance) {
		distToWall = minCB.y+forceDistance - p.position_.y;
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(0, 1, 0) * factor * 2.0*forceStrength;
	} else if (p.position_.y > maxCB.y-forceDistance) {
		distToWall = p.position_.y - (maxCB.y-forceDistance);
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(0, -1, 0) * factor * forceStrength;
	}

	// x, z boundaries
	// xy plane
	if (p.position_.x < minCB.x+forceDistance) {
		distToWall = minCB.x+forceDistance - p.position_.x;
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(1, 0, 0) * factor * forceStrength;
	}
	if (p.position_.x > maxCB.x-forceDistance) {
		distToWall = p.position_.x - (maxCB.x-forceDistance);
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(-1, 0, 0) * 1 * factor * forceStrength;   // boundary force is way too weak ****
	}		

	// yz plane
	if (p.position_.z < minCB.z+forceDistance) {
		distToWall = minCB.z+forceDistance - p.position_.z;
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(0, 0, 1) * factor * forceStrength;
	} 
	if (p.position_.z > maxCB.z-forceDistance) {
		distToWall = p.position_.z - (maxCB.z-forceDistance);
		factor = distToWall * invForceDist;
		force += vmml::Vector3f(0, 0, -1) * factor * forceStrength;
	}

	return force;
}

void UnifiedPhysics::BoundaryHandlingBoxSimple(const int i)
{
	// simple collision handling with domain boundary for fluid particles

	static const vmml::Vector3f& minBox = fc_->collisionBox.getMin();
	static const vmml::Vector3f& maxBox = fc_->collisionBox.getMax();

	static const float damping = 0.9f; // 0.1
	// particle-plane collision v_reflect = v_orgin - N*dot(N, v)* reflect
	// elastic collision / soft collision, value between 1.0 and 2.0.
	static const float reflect = 1.1f; //1.5;  //2.0 zero damping, 1.0 damping maximal 

	UnifiedParticle& p = particles_[i];

	//* if particle is not a frozen particle
	if(p.type_ != FROZEN_PARTICLE)
	{
		for (int j = 0; j < 3; j++) {
			// check minimum box boundary
			if (p.position_[j] < minBox[j]) {
				p.position_[j] = minBox[j];
				p.velocity_[j] *= damping;
			}
			// check maximum box boundary
			if (p.position_[j] > maxBox[j]) {
				p.position_[j] = maxBox[j];
				p.velocity_[j] *= damping;
			}
		}
	}

	//*/

	/* if particle is not a frozen particle
	// TODO: problem exists here -> particles would stick to boundaries (espetially front wall & right wall) to some extent
	if(p.type != FROZEN_PARTICLE)
	{
		// collision with ground
		if (p.position[1] < minBox[1])
		{
			p.position[1] = minBox[1];
			vmml::Vector3f axis(0, 1, 0);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[1] *= damping;
		}

		// collision with ceiling
		if (p.position[1] > maxBox[1])
		{
			p.position[1] = maxBox[1];
			vmml::Vector3f axis(0, -1, 0);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[1] *= damping;
		}

		// collision with left wall
		if (p.position[0] < minBox[0])
		{
			p.position[0] = minBox[0];
			vmml::Vector3f axis(1, 0, 0);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[0] *= damping;
		}

		// collision with right wall
		if (p.position[0] > maxBox[0])
		{
			p.position[0] = maxBox[0];
			vmml::Vector3f axis(-1, 0, 0);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[0] *= damping;
		}

		// collision with back wall
		if (p.position[2] < minBox[2])
		{
			p.position[2] = minBox[2];
			vmml::Vector3f axis(0, 0, 1);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[2] *= damping;
		}

		// collision with front wall
		if (p.position[2] > maxBox[2])
		{
			p.position[2] = maxBox[2];
			vmml::Vector3f axis(0, 0, -1);
			p.velocity -= axis * axis.dot(p.velocity[1]) * reflect;
			p.velocity[2] *= damping;
		}	
	}//*/

	// get new spatial index
	p.index_ = zIndex.CalcIndex(fc_->scales * p.position_);  
}

void UnifiedPhysics::BoundaryHandlingBoxPerParticleOMP()
{
	const vmml::Vector3f minBox = fc_->collisionBox.getMin();
	const vmml::Vector3f maxBox = fc_->collisionBox.getMax();
	static const float damping = -0.5; // 0.1
	static const int numParticles = particles_.size(); 

	int i;
	int chunk = 100;
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numParticles; i++)
		{
			UnifiedParticle* p = &particles_[i];
			// we only do collision handling for non-frozen particles
			if(p->type_ != FROZEN_PARTICLE)
			{
				RigidBody* parentBody = p->parent_rigidbody_;
				if(!parentBody)
				{
					// non rigid & non boundary particles
					BoundaryHandlingBoxPerNonRigidParticle(minBox, maxBox, damping, p->position_, p->velocity_);
				}
			}

			// get new spatial index
			p->index_ = zIndex.CalcIndex(fc_->scales * p->position_);  
		}
	}
}

void UnifiedPhysics::BoundaryHandlingPureFluidOMP()
{
	const vmml::Vector3f minBox = fc_->collisionBox.getMin();
	const vmml::Vector3f maxBox = fc_->collisionBox.getMax();
	static const float damping = -0.5; // 0.1
	static const int numParticles = particles_.size(); 

	int i;
	int chunk = 100;
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numParticles; i++)
		{
			UnifiedParticle* p = &particles_[i];

			if(p->type_ == LIQUID_PARTICLE)
			{
				// non rigid & non boundary particles
				BoundaryHandlingBoxPerNonRigidParticle(minBox, maxBox, damping, p->position_, p->velocity_);
			}

			// get new spatial index
			p->index_ = zIndex.CalcIndex(fc_->scales * p->position_);  
		}
	}
}

void UnifiedPhysics::BoundaryHandlingBoxPerNonRigidParticle(const vmml::Vector3f& minBox, const vmml::Vector3f& maxBox, const float damping, vmml::Vector3f& pos, vmml::Vector3f& vel)
{
	// ground
	if (pos[1] < minBox[1]) 
	{
		pos[1] =  minBox[1];
		vel[1] *= damping;
	}

	// ceiling
	if (pos[1] > maxBox[1]) 
	{
		pos[1] =  maxBox[1];
		vel[1] *= damping;
	}

	// left wall
	if (pos[0] < minBox[0]) 
	{
		pos[0] = minBox[0];
		vel[0] *= damping;
	}

	// right wall
	if (pos[0] > maxBox[0]) 
	{
		pos[0] =  maxBox[0];
		vel[0] *= damping;
	}

	// back wall
	if (pos[2] < minBox[2]) 
	{
		pos[2] = minBox[2];
		vel[2] *= damping;
	}

	// front wall
	if (pos[2] > maxBox[2]) 
	{
		pos[2] = maxBox[2];
		vel[2] *= damping;
	}

}

void UnifiedPhysics::ApplyRotationToParticlesHost(RigidBody& body)
{

}

void UnifiedPhysics::CudaParticlePhysics()
{
	if(dptr_.particleCount > 0 )
	{


#ifdef USE_VBO_CUDA
		// (1) Map
		// -> Context switching OPENGL -> CUDA
		for (int i = 0; i < 2; ++i)
		{
			dptr_.d_pos_zindex[i] = (float4 *) MapGLBufferObject(&cuda_posvbo_resource_[i]);  
		}

		// (2) DoCUDAwork
		switch(fc_->physicsType)
		{
		case 'o': // original SPH
		case 'w': // WCSPH
			// compute physics in parallel using CUDA
			//ParticlePhysicsOnDeviceSPH(dptr_);
			//ParticlePhysicsOnDeviceSPHApproximate(dptr_);
			ParticlePhysicsOnDeviceFluidRigidCouplingSPH(dptr_);
			break;

		case 'i': // incompressible SPH (PCISPH)
			//ParticlePhysicsOnDevicePCISPH(dptr_);							// Pure incompressible fluids with simple boundaries
			//ParticlePhysicsOnDevicePCISPHIhmsen2010(dptr_);				// Fluids with Static Complex Boundaries (Ihmsen2010's method)
			//ParticlePhysicsOnDeviceFluidStaticBoundariesPCISPH(dptr_);		// Fluids with Static Complex Boundaries
			ParticlePhysicsOnDeviceFluidRigidCouplingPCISPH(dptr_);		// Two-way coupling of incompressible Fluids and Rigid Bodies
			break;

		}

		// (3) Unmap
		// note: do unmap after cudaWork() here to avoid unnecessary graphics/CUDA context switch
		// -> Context switching CUDA -> OPENGL
		for (int i = 0; i < 2; ++i)
		{
			UnmapGLBufferObject(cuda_posvbo_resource_[i]);
		}

		// (4) buffer swap
		// switch ping pong buffers
		// we do this swap after unmap
		std::swap(dptr_.m_posVbo[0], dptr_.m_posVbo[1]);	// for vbo swap

		if (renderer_)
		{
			renderer_->setVertexBuffer(GetCurrentReadBuffer(), particles_.size());
		}

#else

		switch(fc_->physicsType)
		{
		case 'o': // original SPH
		case 'w': // WCSPH
			// compute physics in parallel using CUDA
			//ParticlePhysicsOnDeviceSPH(dptr_);
			//ParticlePhysicsOnDeviceSPHApproximate(dptr_);
			ParticlePhysicsOnDeviceFluidRigidCouplingSPH(dptr_);
			break;

		case 'i': // incompressible SPH (PCISPH)
			//ParticlePhysicsOnDevicePCISPH(dptr_);							// Pure incompressible fluids with simple boundaries
			//ParticlePhysicsOnDevicePCISPHIhmsen2010(dptr_);				// Fluids with Static Complex Boundaries (Ihmsen2010's method)
			//ParticlePhysicsOnDeviceFluidStaticBoundariesPCISPH(dptr_);		// Fluids with Static Complex Boundaries
			ParticlePhysicsOnDeviceFluidRigidCouplingPCISPH(dptr_);		// Two-way coupling of incompressible Fluids and Rigid Bodies
			break;

		}

		memset(particle_info_for_rendering_.p_pos_zindex,			0,		dptr_.particleCountRounded * sizeof(float4) );
		memset(particle_info_for_rendering_.p_vel,					0,		dptr_.particleCountRounded * sizeof(float4) );
		memset(particle_info_for_rendering_.p_corr_pressure,		0,		dptr_.particleCountRounded * sizeof(float) );
		memset(particle_info_for_rendering_.p_predicted_density,	0,		dptr_.particleCountRounded * sizeof(float) );
		memset(particle_info_for_rendering_.p_type,					0,		dptr_.particleCountRounded * sizeof(int) );
		memset(particle_info_for_rendering_.p_activeType,			0,		dptr_.particleCountRounded * sizeof(int) );
		CopyGPUParticles(particle_info_for_rendering_);

#endif

#ifdef DUMP_PARTICLES_TO_FILE

		DumpGPUParamsToFile();

#endif
	}

}

void UnifiedPhysics::CpuParticlePhysics()
{
	switch(fc_->physicsType)
	{
	case 'o':	// original SPH
	case 'w':	// WCSPH	
		CpuParticlePhysicsPureFluidsSPH();
		//CpuParticlePhysicsStaticBoundariesSPH();
		//CpuParticlePhysicsBoundaryForceMonaghan();				// TODO: incorrect
		//CpuParticlePhysicsWithOneWaySolidToFluidCoupling();		// TODO: incorrect
		//CpuParticlePhysicsSimpleRigidFluidCoupling();
		break;
	case 'i':
		//cpuParticlePhysicsPureFluidsPCISPH();
		CpuParticlePhysicsVersatileCouplingPCISPH();	// cpu version of two-way coupling of incompressible fluids and rigid bodies
		//CpuParticlePhysicsStaticBoundariesPCISPH();	// our static boundaries
		break;

	}
}

vmml::Vector3f UnifiedPhysics::CalculateSurfaceTensionForcePCISPH(const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, const float splineCoefficient)
{
	vmml::Vector3f force = (p_pos - neigh_pos) * fc_->surface_tension_gamma * fc_->initialMass * fc_->initialMass * splineCoefficient * (1.0f/dist);

	return force;
}

vmml::Vector3f UnifiedPhysics::CalculateSurfaceCohesionForcePCISPH(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, float neigh_weighted_vol, WeightingKernel& weightingKernel)
{
	float splineCoefficient = weightingKernel.lut_spline_surface_adhesion()[dist_lut];

	vmml::Vector3f force = (p_pos - neigh_pos) * fc_->surface_adhesion_beta * fc_->initialMass * fc_->fluidRestDensity * neigh_weighted_vol * splineCoefficient * (1.0f/dist); 

	return force;
}

void UnifiedPhysics::CpuParticlePhysicsStaticBoundariesSPH()
{
	NeighborSearchOMP();

	// for each liquid particle, we compute its density & pressure using SPH/WCSPH/PCISPH
	CalculateLiquidParticleDensityPressureOMP();

	// for each 
	// (1) liquid particle, we calculate its force from liquid particles, rigid particles and boundaries
	CalculateForcePerParticleWithSimpleBoxCollisionHandlingOMP();

	// TODO: New Idea: Corrected constraint force
	// for each liquid particle, we calculate corrected forces from the updated rigid particles
	// and then we do liquid particle integration
	UpdateLiquidParticlePosVelSphCPUOMP();

	// TODO: There are lots of work to do here. New Ideas???
	// Boundary Handling
	BoundaryHandlingBoxOMP();

	// resort particles array using radix sort using their z-index values
	SortParticles();
}

void UnifiedPhysics::CpuParticlePhysicsPureFluidsPCISPH()
{
	NeighborSearchOMP();

	CalculateExternalForcesPCISPHOMP();

	PredictionCorrectionStep();

	UpdateLiquidParticlePosVelCPUOMP();

	BoundaryHandlingPureFluidOMP();

	// resort particles array using radix sort using their z-index values
	SortParticles();
}

void UnifiedPhysics::CpuParticlePhysicsStaticBoundariesPCISPH()
{
	NeighborSearchOMP();

	CalculateWeightedVolumeOMP();

	CalculateExternalForcesStaticBoundariesPCISPHOMP();

	PredictionCorrectionStep();

	UpdateLiquidParticlePosVelSphCPUOMP();

	// TODO: There are lots of work to do here. New Ideas???
	// Boundary Handling
	BoundaryHandlingBoxOMP();

	// resort particles array using radix sort using their z-index values
	SortParticles();
}

vmml::Vector3f UnifiedPhysics::CalculateBoundaryFluidPressureForceHost(uint dist_lut, vmml::Vector3f p_pos, vmml::Vector3f neigh_pos, float p_density, float p_corr_pressure, float weighted_vol)
{
	vmml::Vector3f force(0.0f, 0.0f, 0.0f);
	float  p_inv_den  = 1.0f / p_density;  

	vmml::Vector3f kernel_gradient = ( p_pos - neigh_pos ) * my_kernel_->kernelPressureGradLut(dist_lut);
	float press_grad = p_corr_pressure * p_inv_den * p_inv_den;			// TODO: test if we could use p_inv_den * p_inv_den instead of invfluidRestDensity * invfluidRestDensity in here
	force = kernel_gradient * press_grad * fc_->initialMass *fc_->fluidRestDensity * weighted_vol * (-1.0f);		// Note: don't forget the force is negative

	return force;
}

void UnifiedPhysics::CpuParticlePhysicsPureFluidsSPH()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;


	// -----------------------------Neighbor Search-----------------------------
#pragma omp parallel default(shared) private(i)
	{	  	
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			GetNeighbors(i);
	}


	// -----------------------------Density & Pressure Calculation-----------------------------
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateDensityPressurePureFluidSPH(i);
	}


	// -----------------------------Force Calculation-----------------------------
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
	for (i = 0; i < particles_.size(); i++)
	{
		UnifiedParticle &p = particles_[i];
		if (p.type_ == LIQUID_PARTICLE) // only calculate forces for non frozen particles
		{
			p.force_ = CalculateLiquidParticleForces(p, i);
			// add gravity force
			p.force_.y -= fc_->initialMass * fc_->gravityConst; 
		}
	}


	// -----------------------------Time Integration-----------------------------
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}


	// -----------------------------Boundary Handling-----------------------------
	const vmml::Vector3f minBox = fc_->collisionBox.getMin();
	const vmml::Vector3f maxBox = fc_->collisionBox.getMax();
	static const float damping = -0.5; // 0.1
	static const int numParticles = particles_.size(); 

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numParticles; i++)
		{
			UnifiedParticle* p = &particles_[i];

#ifdef USE_DEBUG
			// we only do collision handling for non-frozen particles
			assert(p->type_ == LIQUID_PARTICLE);
#endif
			// non rigid & non boundary particles
			BoundaryHandlingBoxPerNonRigidParticle(minBox, maxBox, damping, p->position_, p->velocity_);

			// get new spatial index
			p->index_ = zIndex.CalcIndex(fc_->scales * p->position_);  
		}
	}


	// -----------------------------resort particles array using radix sort using their z-index values-----------------------------
	SortParticles();
}

void UnifiedPhysics::CpuParticlePhysicsSimpleRigidFluidCoupling()
{
	NeighborSearchOMP();

	// Melting & Freezing
	//MeltingFreezingOMP();
	
	// for each liquid particle, we compute its density & pressure using SPH/WCSPH/PCISPH
	CalculateLiquidParticleDensityPressureOMP();

	// for each 
	// (1) liquid particle, we calculate its force from liquid particles, rigid particles and boundaries
	// (2) rigid particle, we calculate its force from liquid particles, rigid particles and boundaries 
	CalculateForcePerParticleWithSimpleBoxCollisionHandlingOMP();

	// for each RigidBody, we calculate its force & torque, then linear & angular momenta, and then we do integration
	CalculateRigidBodyMomentaOMP();

	// for each RigidBody, we synchronize rigid particle values according to the rigid body it's part of	
	SynRigidParticlesOMP();

	// TODO: New Idea: Corrected constraint force
	// for each liquid particle, we calculate corrected forces from the updated rigid particles
	// and then we do liquid particle integration
	UpdateLiquidParticlePosVelSphCPUOMP();

	// TODO: There are lots of work to do here. New Ideas???
	// Boundary Handling
	BoundaryHandlingBoxOMP();

	// resort particles array using radix sort using their z-index values
	SortParticles();

	UpdateRigidBodyParticleInformationOMP();
}

void UnifiedPhysics::CalculateForcePerParticleWithSimpleBoxCollisionHandlingOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	switch(fc_->physicsType)
	{
		//-----------------------------------------------------------------------------------
		//-----------------------------original SPH & WCSPH----------------------------------
		//-----------------------------------------------------------------------------------
	case 'o':
	case 'w':
		//*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateForcesPerParticleStep1(i);
		//*/

		/*
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep2(i);
		//*/

		/*
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep3ForElasticBodies(i);
		//*/

		/*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep3ForElasticBodies(i);
		//*/
		break;

	case 'i': 
		//-----------------------------------------------------------------------------------
		//--------------------------incompressible SPH (PCISPH)------------------------------
		//-----------------------------------------------------------------------------------
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateExternalForcesPCISPH(i);

		PredictionCorrectionStep();
		break;
	}

}

void UnifiedPhysics::CalculateExternalForcesPCISPHOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
	for (i = 0; i < particles_.size(); i++)
		CalculateExternalForcesPCISPH(i);
}

void UnifiedPhysics::CalculateExternalForcesStaticBoundariesPCISPHOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
	for (i = 0; i < particles_.size(); i++)
		CalculateExternalForcesStaticBoundariesPCISPH(i);

}

void UnifiedPhysics::CalculateExternalForcesVersatilCouplingPCISPHOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
	for (i = 0; i < particles_.size(); i++)
		CalculateExternalForcesVersatilCouplingPCISPH(i);
}

void UnifiedPhysics::UpdateLiquidParticlePosVelCPUOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}
}

void UnifiedPhysics::NeighborSearchOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{	  
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			GetNeighbors(i);
	}
}

void UnifiedPhysics::CalculateWeightedVolumeOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateWeightedVolume(i);
	}	
}

void UnifiedPhysics::CorrectedDensityPressureComputationOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	// foreach liquid particle, we compute its density & pressure using WCSPH
	// TODO: Incorporate PCISPH
	//*
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateCorrectedDensitiesPressureVersatileCoupling(i);
			//CalculateDensityPressure(i);
	}
	//*/
}

void UnifiedPhysics::ForceComputationVersatileCouplingOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	// foreach liquid particle, we add its forces from both inside liquids & from boundary particles
	switch(fc_->physicsType)
	{
		//-----------------------------------------------------------------------------------
		//-----------------------------original SPH & WCSPH----------------------------------
		//-----------------------------------------------------------------------------------
	case 'o':
	case 'w':
		//*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateParticleForceVersatilCoupling(i);
			//CalculateForcesPerParticleStep1(i);
		//*/
		break;

	case 'i': 
		//-----------------------------------------------------------------------------------
		//--------------------------incompressible SPH (PCISPH)------------------------------
		//-----------------------------------------------------------------------------------
		// TODO: to be modified
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			//calculateExternalForcesPCISPH(i);
			CalculateForcesISPHVersatilCoupling(i);

		PredictionCorrectionStepVersatileCoupling();
		break;
	}
}

void UnifiedPhysics::UpdateLiquidParticlePosVelSphOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

		//*
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}
	//*/

	/************************************************************************/
	/*                collision handling for each particle                  */
	/************************************************************************/
	//* TODO: Is this still good for rigid particles???
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			BoundaryHandlingBoxSimple(i);
	}
	//*/
}

void UnifiedPhysics::RigidBodyIntegrationOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	/************************************************************************/
	/*                         Integrate Rigid Bodies                       */
	/************************************************************************/
	//*
	const unsigned int numRigidBodies = rigid_bodies_.size();
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numRigidBodies; i++)
			CalculateRigidBodyForceTorqueMomenta(i);
	}
	//*/
}

void UnifiedPhysics::MeltingFreezingOMP()
{
	//UpdateStateOfAggregation();

	/* only if heating is on
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;
	if(fc->heatingGround)
	{
#pragma omp parallel default(shared) private(i)
		{
#pragma omp for schedule(dynamic, chunk) nowait
			for (i = 0; i < particles.size(); i++)
				HeatingFluid(i);
		}
	}
	//*/
}

void UnifiedPhysics::CpuParticlePhysicsBoundaryForceMonaghan()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	/************************************************************************/
	/*                         Neighbor Search                              */
	/************************************************************************/
	// TODO: applying activated boundary particles for performance
#pragma omp parallel default(shared) private(i)
	{	  
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			GetNeighbors(i);
	}

	// stats/debugging
	min_dens_ = 10e10;
	max_dens_ = 0.0;
	float densAvgTimestep = 0;


	/************************************************************************/
	/*                         Melting & Freezing                           */
	/************************************************************************/
	//UpdateStateOfAggregation();

	/* only if heating is on
	if(fc->heatingGround)
	{
#pragma omp parallel default(shared) private(i)
		{
#pragma omp for schedule(dynamic, chunk) nowait
			for (i = 0; i < particles.size(); i++)
				HeatingFluid(i);
		}
	}
	//*


	/************************************************************************/
	/*                   Density & Pressure Calculation                     */
	/************************************************************************/

	// foreach liquid particle, we compute its density & pressure using WCSPH
	// TODO: Incorporate PCISPH

	//*
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateDensityPressure(i);
	}
	//*/

	/* debugging
	std::cout << "minimum density in iteration " << iterationCounter << ": " << min_dens_ << std::endl;
	std::cout << "maximum density in iteration " << iterationCounter << ": " << max_dens_ << std::endl << std::endl;
	//*/


	/************************************************************************/
	/*                        Liquid Force Calculation                      */
	/************************************************************************/
	// foreach liquid particle, we add its forces from both inside liquids & from boundary particles
	switch(fc_->physicsType)
	{
		//-----------------------------------------------------------------------------------
		//-----------------------------original SPH & WCSPH----------------------------------
		//-----------------------------------------------------------------------------------
	case 'o':
	case 'w':
		//*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateBoundaryForceMonaghan(i);		// TODO: incorrect boundary force calculation
		//*/

		break;

	case 'i': 
		//-----------------------------------------------------------------------------------
		//--------------------------incompressible SPH (PCISPH)------------------------------
		//-----------------------------------------------------------------------------------
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateExternalForcesPCISPH(i);

		PredictionCorrectionStep();
		break;
	}

	/************************************************************************/
	/*                         Integrate Rigid Bodies                       */
	/************************************************************************/
	/*
	const unsigned int numRigidBodies = rigidBodies.size();
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numRigidBodies; i++)
			CalculateRigidBodyForceTorqueMomenta(i);
	}
	//*/

	//*
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}
	//*/


	/************************************************************************/
	/*                collision handling for each particle                  */
	/************************************************************************/
	/* TODO: Is this still good for rigid particles???
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles.size(); i++)
			BoundaryHandlingBoxSimple(i);
	}
	//*/


	/************************************************************************/
	/*                      re-sort based on spatial index                  */
	/************************************************************************/
	SortParticles();


	/************************************************************************/
	/*                      Update Rigid Body Particles                     */
	/************************************************************************/
	// foreach moving rigid body, we synchronize its boundary particles
	UpdateRigidBodyParticleInformationOMP();

}

void UnifiedPhysics::CalculateLiquidParticleDensityPressureOMP()
{
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateDensityPressure(i);
	}
}

void UnifiedPhysics::CalculateRigidBodyMomentaOMP()
{
	//*
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	const unsigned int numRigidBodies = rigid_bodies_.size();
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numRigidBodies; i++)
			CalculateRigidBodyForceTorqueMomenta(i);
	}
	//*/
}

void UnifiedPhysics::SynRigidParticlesOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < rigid_bodies_.size(); i++)
			SynRigidParticlesPerRigidBody(rigid_bodies_[i]);
	}
}

void UnifiedPhysics::UpdateLiquidParticlePosVelSphCPUOMP()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles_.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}
}

void UnifiedPhysics::BoundaryHandlingBoxOMP()
{
	BoundaryHandlingBoxPerParticleOMP();			
}

void UnifiedPhysics::CpuParticlePhysicsWithOneWaySolidToFluidCoupling()
{
	// compute physics in parallel using OpenMP
	int i;
	int chunk = 100;

	/************************************************************************/
	/*                         Neighbor Search                              */
	/************************************************************************/
#pragma omp parallel default(shared) private(i)
	{	  	
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			GetNeighbors(i);
	}

	// stats/debugging
	min_dens_ = 10e10;
	max_dens_ = 0.0;
	float densAvgTimestep = 0;


	/************************************************************************/
	/*                         Melting & Freezing                           */
	/************************************************************************/
	//UpdateStateOfAggregation();

	/* only if heating is on
	if(fc->heatingGround)
	{
#pragma omp parallel default(shared) private(i)
		{
#pragma omp for schedule(dynamic, chunk) nowait
			for (i = 0; i < particles.size(); i++)
				HeatingFluid(i);
		}
	}
	//*/


	/************************************************************************/
	/*                   Density & Pressure Calculation                     */
	/************************************************************************/

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateDensityPressure(i);
	}


	/************************************************************************/
	/*                         Force Calculation                          */
	/************************************************************************/
	switch(fc_->physicsType)
	{
		//-----------------------------------------------------------------------------------
		//-----------------------------original SPH & WCSPH----------------------------------
		//-----------------------------------------------------------------------------------
	case 'o':
	case 'w':
		//*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateForcesWithoutBoundaryControllingForces(i);
		//*/

		/*
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep2(i);
		//*/

		/*
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep3ForElasticBodies(i);
		*/
		/*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles.size(); i++)
			CalculateForcesPerParticleStep3ForElasticBodies(i);
		//*/

		//*
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			PredictIntegrationOneWaySolidToFluidCoupling(i);

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			DetectCorrectFluidRigidCollision(i);
		//*/

		break;

	case 'i': 
		//-----------------------------------------------------------------------------------
		//--------------------------incompressible SPH (PCISPH)------------------------------
		//-----------------------------------------------------------------------------------
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < particles_.size(); i++)
			CalculateExternalForcesPCISPH(i);

		PredictionCorrectionStep();
		break;
	}

	/************************************************************************/
	/*                         Integrate Rigid Bodies                       */
	/************************************************************************/
	/*
	const unsigned int numRigidBodies = rigidBodies.size();
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < numRigidBodies; i++)
			CalculateRigidBodyForceTorqueMomenta(i);
	}
	//*/

	/*

#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles.size(); i++)
			UpdateLiquidParticlePosVelSphCPU(i);
	}
	//*/


	/************************************************************************/
	/*                collision handling for each particle                  */
	/************************************************************************/
	/* TODO: Is this still good for rigid particles???
#pragma omp parallel default(shared) private(i)
	{
#pragma omp for schedule(dynamic, chunk) nowait	  
		for (i = 0; i < particles.size(); i++)
			BoundaryHandlingBoxSimple(i);
	}
	//*/


	/************************************************************************/
	/*                      re-sort based on spatial index                  */
	/************************************************************************/
	SortParticles();


	/************************************************************************/
	/*                      Update Rigid Body Particles                     */
	/************************************************************************/
	UpdateRigidBodyParticleInformationOMP();

}

void UnifiedPhysics::CpuParticlePhysicsVersatileCouplingPCISPH()
{
	// neigh search
	NeighborSearchOMP();

	// calculate weighted volume
	//CalculateWeightedVolumeOMP();

	// compute density, pressure and forces for each liquid particle & rigid particle
	CalculateExternalForcesVersatilCouplingPCISPHOMP();

	PredictionCorrectionStep();

	// Integrate liquid particles
	UpdateLiquidParticlePosVelSphCPUOMP();

	BoundaryHandlingPureFluidOMP();

	// Three steps:
	//	(1)force & torque calculation
	//	(2)linear & angular momentum calculation
	//	(3)Rigid Body Integration
	RigidBodyIntegrationOMP();

	// resort particles array using radix sort using their z-index values
	SortParticles();

	// Synchronize rigid particles
	UpdateRigidBodyParticleInformationOMP();

	// for each RigidBody, we synchronize rigid particle values according to the rigid body it's part of	
	SynRigidParticlesOMP();

}

void UnifiedPhysics::CalculateForcesISPHVersatilCoupling(const int i)
{	
	// this method is called when using PCISPH
	UnifiedParticle &p = particles_[i];
	std::vector<int> &indices = neighbor_indices_[i];

	p.force_ = (0.0, 0.0, 0.0);
	const float restVolume = fc_->initialMass  / fc_->fluidRestDensity;

	for (int j = 0; j < indices.size(); j++) {
		UnifiedParticle& neigh = particles_[indices[j]];
		float dist = p.position_.distance(neigh.position_);
		if (dist < fc_->globalSupportRadius) {
			/*
			// real neighbors
			const vmml::Vector3f x_ij = p.position - neigh.position;
			const vmml::Vector3f v_ij = p.velocity - neigh.velocity;
			const vmml::Vector3f kernelGradient = x_ij * myKernel->kernelPressureGradLut(dist); // symmetric
			const vmml::Vector3f kernelViscosity = x_ij * myKernel->kernelViscosityLaplacianLut(dist); // symmetric

			if (neigh.type == LIQUID_PARTICLE)
			{	
				// ---------------------------versatile methods: Problem exists-----------------------------------------------
				// sum up viscosity force from liquid particles according to E.q(11)
				// negative symmetry
				float numerator = MAX(0.0, x_ij.dot(v_ij)); 
				float denominator = x_ij.lengthSquared() + fc->epsilon_h_square;
				float pi = -1.0f * fc->nu_liquid * numerator / denominator;
				p.force -= kernelViscosity * fc->initialMass * fc->initialMass * pi;
				//
			}
			else if (neigh.type == FROZEN_PARTICLE)
			{
				// ---------------------------versatile methods: Problem exists-----------------------------------------------
				// sum up viscosity force from boundary particles according to E.q(13)
				float numerator = MAX(0.0, x_ij.dot(v_ij)); 
				float denominator = x_ij.lengthSquared() + fc->epsilon_h_square;
				float pi = -1.0f * fc->nu_rigid_liquid * numerator / denominator;
				p.force -= kernelViscosity * fc->initialMass * fc->fluidRestDensity * neigh.weighted_volume * pi;
				//
			}	
			//*/

			//*
			float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric

			// compute artificial viscosity according to MCG03
			// negative symmetry
			vmml::Vector3f v_ij = p.velocity_ - particles_[indices[j]].velocity_;
			p.force_ -= v_ij * restVolume * restVolume * fc_->fluidViscConst * kernelVisc;
			//*/

		}
	}

	// add gravity
	p.force_.y -= fc_->initialMass * fc_->gravityConst; 

	// add boundary forces 
	if (fc_->addBoundaryForce)
		p.force_ += BoxBoundaryForce(i);

	// init some quantities which are going to be used in the prediction step	
	p.correction_pressure_ = 0.0f;
	p.correction_pressure_force_ = (0.0, 0.0, 0.0);

	// small modification of the pressure-density(Gas/Tait) equation
	//const float p_0 = fc->fluidRestDensity * fc->gravityConst * fc->fluidHeight;	// p_0 = rho_0 * g * h
	//p.correctionPressure = p_0; //98100.0f;	// p = p_0 + k(rho - rho_0) & p_0 = rho_0 * g * h
	

}

void UnifiedPhysics::ComputeCorrectivePressureForceVersatilCoupling(const int i)
{
	// init the pressure forces (after checking loop criterion / return above)
	UnifiedParticle& p = particles_[i];
	if (p.type_ != FROZEN_PARTICLE)
	{
		std::vector<int> &indices = neighbor_indices_[i];

		p.correction_pressure_force_ = (0.0, 0.0, 0.0);

		// compute pressure forces with correction pressure
		float densSq = p.density_*p.density_;
		float p_i = p.correction_pressure_;

		const float invDensity = 1.0 / p.density_;
		const float pVol = fc_->initialMass * invDensity;

		for (int j = 0; j < indices.size(); j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				/*
				// real neighbors
				const vmml::Vector3f x_ij = p.position - neigh.position;
				const vmml::Vector3f v_ij = p.velocity - neigh.velocity;
				const vmml::Vector3f kernelGradient = x_ij * myKernel->kernelPressureGradLut(dist); // symmetric
				const vmml::Vector3f kernelViscosity = x_ij * myKernel->kernelViscosityLaplacianLut(dist); // symmetric

				// ---------------------------versatile methods: Problem exists-----------------------------------------------
				if (neigh.type == LIQUID_PARTICLE)
				{			
					// versatile methods
					// sum up pressure force from liquid particles according to E.q(8) 
					// negative symmetry			
					float grad = pVol * pVol * p.pressure;
					vmml::Vector3f tempPressureForce = kernelGradient * grad;
					p.correctionPressureForce -= tempPressureForce;
					//
				}
				else if (neigh.type == FROZEN_PARTICLE)
				{
					// versatile methods
					// sum up pressure force from boundary particles according to E.q(9) 
					float grad = fc->initialMass * fc->fluidRestDensity * neigh.weighted_volume * p.pressure * invDensity * invDensity;
					vmml::Vector3f tempPressureForce = kernelGradient * grad;
					p.correctionPressureForce -= tempPressureForce;
					//
				}	
				//*/
				// ---------------------------versatile methods: Problem exists-----------------------------------------------


				//*
				float p_j = neigh.correction_pressure_;
				float kernelGradientValue = my_kernel_->kernelPressureGradLut(dist);
				vmml::Vector3f kernelGradient = (p.position_ - neigh.position_) * kernelGradientValue;

				// sum up pressure force according to Monaghan
				float grad = p_i / densSq + p_j / (neigh.density_*neigh.density_);
				p.correction_pressure_force_ -= kernelGradient * grad * fc_->initialMass * fc_->initialMass;
				//*/
			}
		}
	}
}

void UnifiedPhysics::CalculateExternalForcesPCISPH(const int i)
{	
	// this method is called when using PCISPH
	UnifiedParticle &p = particles_[i];
	std::vector<int> &indices = neighbor_indices_[i];

	p.force_ = (0.0, 0.0, 0.0);
	const float restVolume = fc_->initialMass  / fc_->fluidRestDensity;

	for (int j = 0; j < indices.size(); j++) {
		float dist = p.position_.distance(particles_[indices[j]].position_);
		if (dist < fc_->globalSupportRadius) {
			float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric

			// compute artificial viscosity according to MCG03
			// negative symmetry
			vmml::Vector3f v_ij = p.velocity_ - particles_[indices[j]].velocity_;
			p.force_ -= v_ij * restVolume * restVolume * fc_->fluidViscConst * kernelVisc;
		}
	}

	// add gravity
	p.force_.y -= fc_->initialMass * fc_->gravityConst; 

	// add boundary forces 
	if (fc_->addBoundaryForce)
		p.force_ += BoxBoundaryForce(i);

	// init some quantities which are going to be used in the prediction step	
	p.correction_pressure_ = 0.0f;
	p.correction_pressure_force_ = (0.0, 0.0, 0.0);
}

void UnifiedPhysics::CalculateExternalForcesStaticBoundariesPCISPH(const int i)
{	
	UnifiedParticle &p = particles_[i];

	if (p.type_ == LIQUID_PARTICLE)
	{
		std::vector<int> &indices = neighbor_indices_[i];
		p.force_ = (0.0, 0.0, 0.0);
		const float restVolume = fc_->initialMass  / fc_->fluidRestDensity;

		for (int j = 0; j < indices.size(); j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				uint dist_lut = dist * (WeightingKernel::lut_size()) / fc_->globalSupportRadius;
				if( dist_lut > WeightingKernel::lut_size() )
					dist_lut = WeightingKernel::lut_size();

				if (neigh.type_ == LIQUID_PARTICLE)
				{
					// (1) add viscosity force
					float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
					// compute artificial viscosity according to MCG03
					// negative symmetry
					vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
					p.force_ -= v_ij * restVolume * restVolume * fc_->fluidViscConst * kernelVisc;


					// (2) TODO: surface tension force : cohesion force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
				} 
				else if (neigh.type_ == FROZEN_PARTICLE)
				{
					// For each liquid particle, we add forces from frozen boundary particles
					p.force_ += CalculateBoundaryFluidPressureForceHost(dist_lut, p.position_, neigh.position_, p.predicted_density_, p.previous_correction_pressure_, p.weighted_volume_);

					// TODO: surface tension force : adhesion force according to E.q(6) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
				}
			}
		}

		// add gravity
		p.force_.y -= fc_->initialMass * fc_->gravityConst; 

		// add boundary forces 
		if (fc_->addBoundaryForce)
			p.force_ += BoxBoundaryForce(i);
	}

	p.previous_correction_pressure_ = p.correction_pressure_;
	// init some quantities which are going to be used in the prediction step	
	p.correction_pressure_ = 0.0f;
	p.correction_pressure_force_ = (0.0, 0.0, 0.0);
}

void UnifiedPhysics::CalculateExternalForcesVersatilCouplingPCISPH(const int i)
{	
	UnifiedParticle &p = particles_[i];

	if (p.type_ == LIQUID_PARTICLE || p.type_ == RIGID_PARTICLE)
	{
		std::vector<int> &indices = neighbor_indices_[i];
		p.force_ = (0.0, 0.0, 0.0);
		const float restVolume = fc_->initialMass  / fc_->fluidRestDensity;

		for (int j = 0; j < indices.size(); j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				uint dist_lut = dist * (WeightingKernel::lut_size()) / fc_->globalSupportRadius;
				if( dist_lut > WeightingKernel::lut_size())
					dist_lut = WeightingKernel::lut_size();

				// for liquid particles
				if (p.type_ == LIQUID_PARTICLE)
				{
					if (neigh.type_ == LIQUID_PARTICLE)
					{
						// (1) // For each liquid particle, we add viscosity force from other liquid particles	
						// viscosity force & cohesion force
						float kernelVisc = my_kernel_->kernelViscosityLaplacianLut(dist); // symmetric
						// compute artificial viscosity according to MCG03
						// negative symmetry
						vmml::Vector3f v_ij = p.velocity_ - neigh.velocity_;
						p.force_ -= v_ij * restVolume * restVolume * fc_->fluidViscConst * kernelVisc;

						// (2) TODO: surface tension force : cohesion force according to E.q(1) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
						if (dist >= fc_->globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
						{
							p.force_ -= CalculateSurfaceTensionForcePCISPHHost(dist_lut, dist, p.position_, neigh.position_);
						}
					} 
					else if (neigh.type_ == FROZEN_PARTICLE || neigh.type_ == RIGID_PARTICLE)
					{
						// For each liquid particle, we add forces from rigid particles & frozen boundary particles
						p.force_ += CalculateBoundaryFluidPressureForceHost(dist_lut, p.position_, neigh.position_, p.predicted_density_, p.previous_correction_pressure_, p.weighted_volume_);

						// TODO: surface tension force : adhesion force according to E.q(6) from paper "Versatile Surface Tension and Adhesion for SPH Fluids"
						if (dist >= fc_->globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
						{
							p.force_ -= CalculateSurfaceCohesionForcePCISPHHost(dist_lut, dist, p.position_, neigh.position_, neigh.weighted_volume_);
						}
					}
				} 
				else if (p.type_ == RIGID_PARTICLE)
				{
					if (neigh.type_ == RIGID_PARTICLE)	// TODO: add neigh.type == FROZEN_PARTICLE
					{
						// For each rigid particle, we add forces from rigid particles of other rigid bodies
						// "Real-Time Rigid Body Simulation on GPUs" from GPU Gems 3
						const float overlap = 2.0f * fc_->particleRadius - dist;
						// only calculate forces between rigid particles if they belong to different rigid body
						if (overlap > 0.00001f)	// use 0.00001f to smooth numeric error
						{
							// the original one : f = f_spring + f_damping "Real-Time Rigid Body Interaction" (5.15) & (5.16)
							// here r_ij = p_j - p_i  v_ij = v_j - v_i
							p.force_	+= CalculateSpringForceHost(dist, overlap, neigh.position_ - p.position_) + CalculateDampingForceHost(neigh.velocity_ - p.velocity_);
						}
					} 
					else if (neigh.type_ == LIQUID_PARTICLE)
					{
						// For each rigid particle, we add forces from liquid particles using the latter's corrected pressure
						// versatile method E.q(10)
						p.force_ -= CalculateBoundaryFluidPressureForceHost(dist_lut, neigh.position_, p.position_, neigh.predicted_density_, neigh.correction_pressure_, neigh.weighted_volume_);

						if (dist >= fc_->globalSupportRadius * 0.25f)	// Avoid Division by Zero Errors
						{
							p.force_ += CalculateSurfaceCohesionForcePCISPHHost(dist_lut, dist, neigh.position_, p.position_, neigh.weighted_volume_);
						}
					}				
				}			
			}
		}

		// add gravity force to liquid particles
		// Note: for rigid particles, we do not add gravity force, since the gravitational force exerts no torque on rigid body, we delay adding rigid body's 
		// gravitational force to rigid body force & torque calculation
		if (p.type_ == LIQUID_PARTICLE)
		{
			p.force_.y -= fc_->initialMass * fc_->gravityConst;

			// Handle boundary forces exerted on liquid particles
			p.force_ += CalculateBoundaryForcePerLiquidParticleHost(p.position_);

			// Add other external forces in here
		}
		else if (p.type_ == RIGID_PARTICLE)
		{
			// TODO: implement Wall Weight boundary pressure force / versatile boundary particle pressure force & friction force
			// Handle boundary forces exerted on rigid particles
			p.force_ += CalculateBoundaryForcePerRigidParticleHost(p.position_, p.velocity_);

			// Add other external forces in here
		}
		

/*
		// add boundary forces 
		if (fc->addBoundaryForce)
			p.force += BoxBoundaryForce(i);
*/

	}

	p.previous_correction_pressure_ = p.correction_pressure_;
	// init some quantities which are going to be used in the prediction step	
	p.correction_pressure_ = 0.0f;
	p.correction_pressure_force_ = (0.0, 0.0, 0.0);
}

void UnifiedPhysics::PredictionCorrectionStep()
{
	int i;
	int chunk = 100;
	int particlesSize = particles_.size();

	density_error_too_large_ = true; // loop has to be executed at least once

	int iteration = 0;
	while( (iteration < fc_->minLoops) || ((density_error_too_large_) && (iteration < fc_->maxLoops)) )	
	{
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)
			PredictPositionAndVelocity(i);

		max_predicted_density_ = 0.0; // loop termination criterion

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)
			ComputePredictedDensityAndPressure(i);

		// check loop termination criterion
		float densityErrorInPercent = MAX(0.1f * max_predicted_density_ - 100.0f, 0.0f); // 100/1000 * max_predicted_density_ - 100; 	

		if(fc_->printDebuggingInfo==1)
			std::cout << "ERROR: " << densityErrorInPercent << "%" << std::endl;

		// set flag to terminate loop if density error is smaller than requested
		if(densityErrorInPercent < fc_->maxDensityErrorAllowed) 
			density_error_too_large_ = false; // stop loop

#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait
		for(int i = 0; i < particles_.size(); i++)
			ComputeCorrectivePressureForce(i);

		iteration++;
	}

	// compute and print average and max number of iterations over whole simulation run
	int outCnt;
	if(fc_->printDebuggingInfo==1) outCnt = 1;
	else outCnt = 50;


	/*
	static int iterationPcisphStarted = iterationCounter-1; // needed to compute avgLoops while changing method during simulation
	avg_loops_sim_ += iteration;
	max_loops_sim_ = max(max_loops_sim_, iteration);
	if(iterationCounter % outCnt == 0)
	{
		cout << "nLoops done = " << iteration << " max_loops_sim_ = " << max_loops_sim_ << " avgLoops = " << avg_loops_sim_ / (iterationCounter-iterationPcisphStarted) << endl;
	}
	if(iteration > fc->maxLoops) cout << "maxLoops reached" << endl;
	//*/
}

void UnifiedPhysics::PredictPositionAndVelocity(const int i)
{
	// v' = v + delta_t * a
	// a = F / m
	// v' = v + delta_t * F * V / m
	// v' = v + delta_t * F * 1/density

	// compute predicted position and velocity
	UnifiedParticle &p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		p.predicted_velocity_ = p.velocity_ + (p.force_ + p.correction_pressure_force_) * fc_->deltaT / fc_->initialMass;
		p.predicted_position_ = p.position_ + p.predicted_velocity_ * fc_->deltaT; // always use position at time t

		CollisionHandling(p.predicted_position_, p.predicted_velocity_); 

		// init some quantities which are going to be used in the next step
		p.predicted_density_ = 0.0;

		// check if particle has valid position (==not nan) to detect instabilities
		if(!(p.predicted_position_[0] == p.predicted_position_[0]) ||
			!(p.predicted_position_[1] == p.predicted_position_[1]) ||
			!(p.predicted_position_[2] == p.predicted_position_[2]))
		{
			std::cout << "Particle has invalid predictedPosition!!" << std::endl;
			abort();
		}
	}
}

void UnifiedPhysics::ComputePredictedDensityAndPressure(const int i)
{
	// sph density
	UnifiedParticle& p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		std::vector<int> &indices = neighbor_indices_[i];

		p.predicted_density_ = kernel_self_ * fc_->initialMass;

		// neighborhoods of current positions are used and not of predicted positions
		// this might lead to problems for large velocities
		for (int j = 0; j < indices.size(); j++) {
			// we need to recompute the distance and kernel values using the predicted positions
			float dist = p.predicted_position_.distance(particles_[indices[j]].predicted_position_);
			if (dist < fc_->globalSupportRadius) {
				float kernelValue = my_kernel_->kernelM4Lut(dist); // use LUT to make loop execution faster

				// sph dens, symmetric version
				p.predicted_density_ += kernelValue * fc_->initialMass;
			}
		}

		// compute density error, correct only compression errors
		p.density_error_ = MAX(p.predicted_density_ - fc_->fluidRestDensity, 0.0f);

		// update pressure
		// densityErrorFactor is precomputed and used as a constant (approximation)
		// do not allow negative pressure corrections
		p.correction_pressure_ += MAX(p.density_error_ * fc_->densityErrorFactor, 0.0f);

		// max_predicted_density_ is needed for the loop termination criterion
		// find maximal predicted density of all particles
		max_predicted_density_ = MAX(max_predicted_density_, p.predicted_density_);
	}
}

void UnifiedPhysics::ComputeCorrectivePressureForce(const int i)
{
	// init the pressure forces (after checking loop criterion / return above)
	UnifiedParticle& p = particles_[i];
	if (p.type_ == LIQUID_PARTICLE)
	{
		std::vector<int> &indices = neighbor_indices_[i];

		p.correction_pressure_force_ = (0.0, 0.0, 0.0);

		// compute pressure forces with correction pressure
		float densSq = fc_->fluidRestDensity*fc_->fluidRestDensity;
		float p_i = p.correction_pressure_;

		for (int j = 0; j < indices.size(); j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				float p_j = neigh.correction_pressure_;

				float kernelGradientValue = my_kernel_->kernelPressureGradLut(dist);
				vmml::Vector3f kernelGradient = (p.position_ - neigh.position_) * kernelGradientValue;

				// sum up pressure force according to Monaghan
				float grad = p_i / densSq + p_j / (fc_->fluidRestDensity*fc_->fluidRestDensity);
				p.correction_pressure_force_ -= kernelGradient * grad * fc_->initialMass * fc_->initialMass;
			}
		}
	}
}

//--------------------------------------------------------------------
void UnifiedPhysics::ComputeDensityErrorFactor()
{
	// 1) create a particle block
	CreatePreParticlesISPH();	

	// 2) find the neighbors for each particle
	UpdateNeighbors();

	// 3) compute gradW values
	ComputeGradWValues();

	// 4) find the particle with the maximal number of neighbors (full neighborhood) and
	//	 compute for this particle the densityErrorFactor = -1.0 / (dt^2* V0^2 * ((-sum(gradW) dot sum(gradW)))-(sum(gradW dot gradW))))
	ComputeFactor();	

	// 5) delete all particles
	particles_.clear();
	neighbor_indices_.clear();
	std::cout << "pre-particles cleared" << std::endl;
}

void UnifiedPhysics::ComputeGradWValues()
{
	const int numParticles = particles_.size();
	for (int i = 0; i < numParticles; ++i)
	{
		particles_[i].sum_grad_w_ = (0.0, 0.0, 0.0);
		particles_[i].sum_grad_w_dot_ = 0.0;
	}

	for (int i = 0; i < numParticles; ++i)
	{
		UnifiedParticle& p = particles_[i];
		std::vector<int> &indices = neighbor_indices_[i];

		for (int j = 0; j < indices.size(); j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				// values for pcorr
				vmml::Vector3f gradVect = (p.position_ - neigh.position_) * my_kernel_->kernelPressureGrad(dist, fc_->globalSupportRadius);

				p.sum_grad_w_ += gradVect;
				neigh.sum_grad_w_ -= gradVect;

				p.sum_grad_w_dot_ += gradVect.dot(gradVect);
				neigh.sum_grad_w_dot_ += vmml::Vector3f::dot(gradVect * (-1.0), gradVect* (-1.0));
			}
		} // end over all neighbors		
	} // end over all particles
}

void UnifiedPhysics::CreatePreParticlesISPH()
{
	float jitter = fc_->relativeFluidJitter * fc_->particleSpacing;

	UnifiedParticle prototype;
	prototype.position_ = (0.0, 0.0, 0.0);
	prototype.density_ = fc_->fluidRestDensity;
	prototype.pressure_ = 0.0;

	float transX = 0.0;
	float transY = 0.0; 
	float transZ = 0.0;

	// create particles
	vmml::Vector3f minVec, maxVec;
	if (fc_->particleSpacing == 0.0125)	// sizeFactor / 80.0 = 0.0125;
	{
		vmml::Vector3f minVec(-0.1, 0, -0.1);
		vmml::Vector3f maxVec(0.1, 0.2, 0.1);
	} 
	else
	{
		float scale = fc_->particleSpacing / 0.0125;
		minVec = vmml::Vector3f(-0.1, 0, -0.1) * scale;
		maxVec = vmml::Vector3f(0.1, 0.2, 0.1) * scale;
	}

	BBox tempFluidBox(minVec, maxVec);
	CreatePartilcesCPU(fc_->particleSpacing, jitter, tempFluidBox);
}

void UnifiedPhysics::InitNeighbors(const int numParticles)
{
	// we use a dummy value to make sure the following three vectors have the same size as particles vector does
	const std::vector<int> dummy_vector_int(max_neighs, std::numeric_limits<int>::max());
	const std::vector<vmml::Vector3f> dummy_vector3f(max_neighs, vmml::Vector3f(FLT_MAX, FLT_MAX , FLT_MAX));

	neighbor_indices_.resize(numParticles, dummy_vector_int); // do not forget this vector
	solid_neighs_.resize(numParticles, dummy_vector_int); // do not forget this vector
	solid_neigh_distances_.resize(numParticles, dummy_vector3f); // do not forget this vector

	for (int i = 0; i < numParticles; ++i)
	{
		neighbor_indices_[i].clear();
		solid_neighs_[i].clear();
		solid_neigh_distances_[i].clear();
	}
}

void UnifiedPhysics::UpdateNeighbors()
{
	const int numParticles = particles_.size();

	int i;
	int chunk = 100;
#pragma omp parallel default(shared) private(i)
#pragma omp for schedule(dynamic, chunk) nowait  	
	for (i = 0; i < particles_.size(); i++)
		GetNeighbors(i);	
}

void UnifiedPhysics::ComputeFactor()
{
	int maxNeighborIndex = 0;
	int maxNeighs = 0;
	const int numParticles = particles_.size();

	for (int i = 0; i < numParticles; ++i)
	{
		UnifiedParticle& p = particles_[i];
		std::vector<int> &indices = neighbor_indices_[i];

		int numNeighbors = 0;
		const int indicesSize = indices.size();
		for (int j = 0; j < indicesSize; j++) {
			UnifiedParticle& neigh = particles_[indices[j]];
			float dist = p.position_.distance(neigh.position_);
			if (dist < fc_->globalSupportRadius) {
				++numNeighbors;
			}
		}

		if(numNeighbors > maxNeighs) 
		{
			maxNeighs = numNeighbors;

			// keep tracking of the index of the particle with maximum neighbors
			maxNeighborIndex = i;	
		}
	}

	// get the particle with maximum neighbors
	UnifiedParticle& p = particles_[maxNeighborIndex];

	// compute new pressure due to density error
	float restVol = fc_->initialMass / fc_->fluidRestDensity;
	float preFactor = restVol * restVol * fc_->deltaT_pcisph * fc_->deltaT_pcisph;
	float gradWTerm = (vmml::Vector3f::dot(p.sum_grad_w_ * (-1.0), p.sum_grad_w_) - p.sum_grad_w_dot_);
	float divisor = preFactor * gradWTerm;

	if(divisor == 0) 
	{
		std::cout << "precompute densErrFactor: division by 0" << std::endl;
		exit(0);
	}

	fc_->densityErrorFactor = -1.0 / divisor;
	//fc->densityErrorFactor *= 0.47;
	fc_->densityErrorFactor *= fc_->densityErrorFactorParameter; 

	std::cout << "densityErrorFactor: " << fc_->densityErrorFactor << " maxNeighs: " << maxNeighs << std::endl;
	return; // this has to be done only for one particle with maximal number of neighbors
}

float UnifiedPhysics::BoundaryForceGamma(float normalDistance, float sr)
{
	float s = normalDistance / sr;
	float result;
	if(s <= 0.0f || s >= 2.0f)
	{
		return 0.0f;
	}
	else
	{
		if (s <= 0.66667f)	// 2/3 = 0.66667
		{
			result = 0.66667f;
		} 
		else if (s <= 1.0f)
		{
			result = 2.0f * s - 1.5f * s * s;
		} 
		else
		{
			result = 0.5f * (2.0f - s) * (2.0f - s);
		}		
	}
	float factor;
	if (normalDistance != 0.0f)
	{
		factor = 0.02f * fc_->speedOfSound * fc_->speedOfSound / normalDistance;	// beta = 0.02 * c_s * c_s / y
	}
	return factor * result;
}

float UnifiedPhysics::BoundaryForceChi(float tangentialDistance, float boundaryParticleSpacing)
{
	if (tangentialDistance <= 0.0f || tangentialDistance >= boundaryParticleSpacing)
	{
		return 0.0f;
	} 
	else
	{
		return 1.0f - tangentialDistance / boundaryParticleSpacing;
	}
}

float UnifiedPhysics::BoundaryForceBxy(float tangentialDistance, float normalDistance, float sr, float boundaryParticleSpacing)
{
	return BoundaryForceChi(tangentialDistance, boundaryParticleSpacing) * BoundaryForceGamma(normalDistance, sr);
}

void UnifiedPhysics::CollisionHandling(vmml::Vector3f &pos, vmml::Vector3f &vel)
{
	static const vmml::Vector3f min = fc_->collisionBox.getMin(), max = fc_->collisionBox.getMax();
	static const float damping = 0.0; // 0.1

	// collision handling with domain boundary
	for (int j = 0; j < 3; j++) {
		// check minimum box boundary
		if (pos[j] < min[j]) {
			pos[j] = min[j];
			vel[j] *= damping;
		}
		// check maximum box boundary
		if (pos[j] > max[j]) {
			pos[j] = max[j];
			vel[j] *= damping;
		}
	}
}

void UnifiedPhysics::PredictIntegrationOneWaySolidToFluidCoupling(const int i)
{
	static const float invMass = 1.0 / fc_->initialMass;
	static const float deltaT = fc_->deltaT;
	UnifiedParticle &p = particles_[i];

	// in case of PCISPH add the pressureForce to total force
	if(fc_->physicsType == 'i') 
		p.force_ += p.correction_pressure_force_;

	// particle is not a frozen particle
	// the position of a frozen particle does not change
	if(p.type_ != FROZEN_PARTICLE)
	{
		RigidBody* parentBody = p.parent_rigidbody_;			

		if(parentBody)
		{
			vmml::Vector3f relativePosition = p.position_ - parentBody->old_center_of_mass();
			vmml::Vector3f newRelativePosition = parentBody->rotation_matrix() * relativePosition;
			// v = linear velocity + tangential velocity
			p.predicted_velocity_ = parentBody->velocity() + (parentBody->angular_velocity()).cross(newRelativePosition);
			p.predicted_position_ = parentBody->rigidbody_pos() + newRelativePosition;
		}
		else
		{		
			/************************************************************************/
			/*                    Symplectic Euler Integration                      */
			/************************************************************************/
			//*
			// v^*(t+h) & x^*(t+h)
			// Now we have predicted velocities & positions
			p.predicted_velocity_ = p.velocity_ + p.force_ * invMass * deltaT;
			p.predicted_position_ = p.position_ + p.predicted_velocity_ * deltaT;
			//*/

			/************************************************************************/
			/*                        Leapfrog integration                          */
			/************************************************************************/
			/*	TODO: we need to adapt Leapfrog integration for OneWaySolidToFluidCoupling
			vmml::Vector3f vel_half_previous = p.velocity_leapfrog;								// read old velocity_leapfrog			 v(t-1/2)
			vmml::Vector3f vel_half_next = vel_half_previous + p.force * invMass * deltaT;		// calculate new velocity_leapfrog		 v(t+1/2) = v(t-1/2) + a(t) * dt	
			p.velocity_leapfrog = vel_half_next;											// update new velocity_leapfrog 
			p.velocity = (vel_half_previous + vel_half_next) * 0.5f;						// update new velocity 					 v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5
			p.position += vel_half_next * deltaT;											// update new position					 p(t+1) = p(t) + v(t+1/2) * dt	
			//*/
		}
	}

	// diffusion: Euler step
	/*
	p.temperature += p.deltaTemperaturePerTime * deltaT;
	if(p.temperature > 100)
		p.temperature = 100;
		*/
}

void UnifiedPhysics::DetectCorrectFluidRigidCollision(const int i)
{
	static const float invMass = 1.0 / fc_->initialMass;
	static const float deltaT = fc_->deltaT;
	// calculate the contact point if there is a collision
	UnifiedParticle& p = particles_[i];
	vmml::Vector3f corrected_pos(0.0, 0.0, 0.0);
	if (p.type_ != FROZEN_PARTICLE) // only calculate forces for non frozen particles
	{
		std::vector<int> &indices = neighbor_indices_[i];
		vmml::Vector3f F_i(0.0, 0.0, 0.0);
		// iterate over all potential neighbors
		for (int j = 0; j < indices.size(); j++) 
		{
			UnifiedParticle &neigh = particles_[indices[j]];
			float dist = p.predicted_position_.distance(neigh.predicted_position_);
			// accumulate F_i & corrected_pos when iterating all potential colliding rigid particles
			if (dist < fc_->collisionDistThreshold && neigh.type_ == FROZEN_PARTICLE) 
			{
				// found a collision between rigid particle & fluid particle
				
				// calculate the contact point x_cp
				vmml::Vector3f Xfluid_Xrigid = p.predicted_position_ - neigh.predicted_position_;
				Xfluid_Xrigid.normalize();
				vmml::Vector3f N = Xfluid_Xrigid;	// normal vector from X_rigid to X_fluid
				vmml::Vector3f x_cp = neigh.predicted_position_ + N * fc_->particleRadius;
				
				vmml::Vector3f x_c = GetCenterOfMassStaticBoundaryBox();
				vmml::Vector3f r = x_cp - x_c;

				// calculate cross product matrix of r
				// 0.0      -r.z       r.y
				// r.z      0.0        -r.x
				// -r.y     r.x        0.0
				Matrix3x3 cross_product_matrix_r;
				cross_product_matrix_r.elements[0][0] = 0.0;
				cross_product_matrix_r.elements[0][1] = -r.z;
				cross_product_matrix_r.elements[0][2] = r.y;
				cross_product_matrix_r.elements[1][0] = r.z;
				cross_product_matrix_r.elements[1][1] = 0.0;
				cross_product_matrix_r.elements[1][2] = -r.x;
				cross_product_matrix_r.elements[2][0] = -r.y;
				cross_product_matrix_r.elements[2][1] = r.x;
				cross_product_matrix_r.elements[2][2] = 0.0;

				Matrix3x3 transpose_cross_product_matrix_r = cross_product_matrix_r.GetTransposedMatrix();

				// compute F_i
				vmml::Vector3f temp_vel;
				float temp = p.predicted_velocity_.dot(N);
				if (temp > 0)
				{
					// To avoid sticking, we substitute the boundary condition (8) by (9) see paper "Direct forcing for Lagrangian rigid-fluid coupling"
					vmml::Vector3f predicted_normal_vel = N * p.predicted_velocity_.dot(N);
					vmml::Vector3f predicted_tangential_vel = p.predicted_velocity_ - predicted_normal_vel;
					temp_vel = predicted_tangential_vel * fc_->coefficient_slip + predicted_normal_vel - p.predicted_velocity_;
				}
				else
				{
					vmml::Vector3f predicted_normal_vel = N * p.predicted_velocity_.dot(N);
					vmml::Vector3f predicted_tangential_vel = p.predicted_velocity_ - predicted_normal_vel;
					vmml::Vector3f normal_vel = N * p.velocity_.dot(N);
					temp_vel = predicted_tangential_vel * fc_->coefficient_slip - normal_vel * fc_->coefficient_restitution;
				}

				// F_i = 1.0 / h * (term1 + term2) * temp_vel
				Matrix3x3 term1 = Matrix3x3::UnitMatrix() * (2.0 / p.particle_mass_); 
				Matrix3x3 inv_inertia_tensor; 
				Matrix3x3::CalculateInverse(GetInertiaTensorStaticBoundaryBox(), inv_inertia_tensor);
				Matrix3x3 term2 = transpose_cross_product_matrix_r * inv_inertia_tensor * cross_product_matrix_r;

				F_i += (term1 + term2) * temp_vel * (1.0 / fc_->deltaT);

				corrected_pos += p.normalized_normal_ * p.position_.distance(neigh.position_);	// The second term on the right hand side in E.q(8) from "Boundary handling and adaptive time-stepping for PCISPH"
				
			}
		}

		/************************************************************************/
		/*                    Symplectic Euler Integration                      */
		/************************************************************************/
		// integrate fluid particles using E.q.(10)
		p.velocity_ = p.predicted_velocity_ + F_i * invMass * deltaT;
		p.position_ = p.predicted_position_ + corrected_pos;	// E.q(8) from "Boundary handling and adaptive time-stepping for PCISPH" 
	}
}

void UnifiedPhysics::PrecomputeDistanceFunction()
{
	const float x = fc_->boxLength;
	const float y = fc_->boxHeight;
	const float z = fc_->boxWidth;
	const float start_x = fc_->realBoxContainer.getMin().x; 
	const float start_y = fc_->realBoxContainer.getMin().y;
	const float start_z = fc_->realBoxContainer.getMin().z;
	for (int i = 0; i < lut_size_distance_; ++i)
	{
		float p_x = x * i / lut_size_distance_;
		for (int j = 0; j < lut_size_distance_; ++j)
		{
			float p_y = y * j / lut_size_distance_;
			for (int k = 0; k < lut_size_distance_; ++k)
			{
				float p_z = z * k / lut_size_distance_;
				distance_function_[i][j][k] = DistanceToWall(start_x + p_x, start_y + p_y, start_z + p_z);
			}
		}
	}
}

float UnifiedPhysics::DistanceToWall(const float& x, const float& y, const float& z, vmml::Vector3f& norm)
{
	// To obtain the distance to the wall boundary, we have to compute the distance from each particle to 
	// all the polygons belonging to the wall boundary and select the minimum distance.
	float result = MAX( fc_->boxLength, MAX(fc_->boxHeight, fc_->boxWidth) );
	float dist = 0.0f;
	float temp = 0.0f;
	norm = vmml::Vector3f(0.0f, 1.0f, 0.0f);

	// compare with dist to ground
	dist = MAX((y - fc_->realBoxContainer.getMin().y) , 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(0.0f, 1.0f, 0.0f);
	}
	
	// compare with dist to ceil
	dist = MAX((fc_->realBoxContainer.getMax().y - y), 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(0.0f, -1.0f, 0.0f);
	}

	// compare with dist to left wall
	dist = MAX((x - fc_->realBoxContainer.getMin().x), 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(1.0f, 0.0f, 0.0f);
	}

	// compare with dist to right wall
	dist = MAX((fc_->realBoxContainer.getMax().x - x), 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(-1.0f, 0.0f, 0.0f);
	}

	// compare with dist to back wall
	dist = MAX((z - fc_->realBoxContainer.getMin().z), 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(0.0f, 0.0f, 1.0f);
	}

	// compare with dist to front wall
	dist = MAX((fc_->realBoxContainer.getMax().z - z), 0.0f);
	temp = MIN(result, dist);
	if (temp < result)
	{
		result = temp;
		norm = vmml::Vector3f(0.0f, 0.0f, -1.0f);
	}

	return result;
}

void UnifiedPhysics::PrecomputeWallWeightFunction()
{
	float effectiveDist = fc_->globalSupportRadius - fc_->particleRadius;	// h-r = 4 * r - r = 3*r
	for (int i = 0; i < lut_size_wall_weight_; ++i)
	{
		float distToWall = effectiveDist * i /lut_size_wall_weight_;
		wall_weight_function_[i] = WallWeight(distToWall, effectiveDist);
	}
}

void UnifiedPhysics::TestWallWeightFunction()
{
	//* debugging for distance_function_
	for (int i = 0; i < lut_size_distance_; ++i)
	{
		for (int j = 0; j < lut_size_distance_; ++j)
		{
			for (int k = 0; k < lut_size_distance_; ++k)
			{
				if (distance_function_[i][j][k] > fc_->boxLength || distance_function_[i][j][k] > fc_->boxHeight || distance_function_[i][j][k] > fc_->boxWidth )
				{
					std::cerr << "the value of distance_function_[i][j][k] is incorrect!!!" << std::endl;
					abort();
				}
				
				//std::cout << "distance_function_[" << i << "][" << j << "][" << k << "] = " << distance_function_[i][j][k] << std::endl;
			}		
		}
		
	}
	//*/

	//* debugging for wall_weight_function_
	for (int i = 0; i < lut_size_wall_weight_; ++i)
	{
		std::cout << "wall_weight_function_[" << i << "] = " << wall_weight_function_[i] << std::endl;
	}
	//*/
}

float UnifiedPhysics::WallWeight(float distToWall, float effectiveDist)
{
	if (distToWall >= effectiveDist)
	{
		return 0.0f;
	}
	else
	{
		// first we determine distances to the maximum 10 potential boundary neighs particles' positions (actually 8 is enough)
		// see Figure 2 Wall weight functions from "Improvement in the Boundary Conditions of Smoothed Particle Hydrodynamics"
		float d1 = distToWall + fc_->particleRadius;
		float d2 = d1 + fc_->particleSpacing;
		float d3 = PythagoreanDist(d1, fc_->particleSpacing);	 // symmetric 
		float d4 = PythagoreanDist(d1, fc_->particleSpacing * 2); // symmetric 
		float d5 = PythagoreanDist(d2, fc_->particleSpacing);	 // symmetric 	
		float d6 = PythagoreanDist(d2, fc_->particleSpacing * 2); // symmetric 

		float kernelValue1 = my_kernel_->kernelM4Lut(d1);
		float kernelValue2 = my_kernel_->kernelM4Lut(d2);
		float kernelValue3 = my_kernel_->kernelM4Lut(d3);
		float kernelValue4 = my_kernel_->kernelM4Lut(d4);
		float kernelValue5 = my_kernel_->kernelM4Lut(d5);
		float kernelValue6 = my_kernel_->kernelM4Lut(d6);

		float result = fc_->initialMass * ( kernelValue1 + kernelValue2 + 2 * ( kernelValue3 + kernelValue4 + kernelValue5 + kernelValue6) );
		return result;
	}
	
}

float UnifiedPhysics::GetDistToWallLut(float x, float y, float z)
{
	static float factor_x = 1.0f / fc_->boxLength;
	static float factor_y = 1.0f / fc_->boxHeight;
	static float factor_z = 1.0f / fc_->boxWidth;
	int index_x = x * factor_x * lut_size_distance_;
	int index_y = y * factor_y * lut_size_distance_;
	int index_z = z * factor_z * lut_size_distance_;

	if (index_x >= lut_size_distance_ || index_y >= lut_size_distance_ || index_z >= lut_size_distance_ )
	{
		return 0.0f;
	}
	else
	{
		return distance_function_[index_x][index_y][index_z];
	}
}

float UnifiedPhysics::GetWallWeightFunctionLut(float distToWall)
{
	float effectiveDist = fc_->globalSupportRadius - fc_->particleRadius;	// h-r = 4 * r - r = 3*r
	static float factor = 1.0f / effectiveDist;
	int index = distToWall * factor * lut_size_wall_weight_;

	return (index >= lut_size_wall_weight_) ? 0.0 : wall_weight_function_[index];
}

vmml::Vector3f UnifiedPhysics::GetBoundaryPressureForceWallWeight(const int i)
{
	// TODO: this version is incorrect!!!
	UnifiedParticle &p = particles_[i];
	vmml::Vector3f norm(0.0f);
	vmml::Vector3f force(0.0f);
	const float distToWall = DistanceToWall(p.position_.x, p.position_.y, p.position_.z, norm);
	static const float effectiveDist = fc_->globalSupportRadius - fc_->particleRadius;
	if (distToWall >= effectiveDist)
	{
		return vmml::Vector3f(0.0f, 0.0f, 0.0f);
	}
	else 
	{
		vmml::Vector3f newVelocity = norm * (effectiveDist - distToWall) / fc_->deltaT;		// v' = (x' - x) / t
		vmml::Vector3f acc = (newVelocity - p.velocity_) / fc_->deltaT;							// a = (v'- v) / t
		force = acc * fc_->initialMass;													// F = ma	
	}

	return force;
}

void UnifiedPhysics::DoPhysics()
{

#ifdef SPH_DEMO_SCENE_2

	DoPhysicsDemo2();

#else

	DoPhysicsDemo1();

#endif

	// redraw
	glutPostRedisplay();

}

void UnifiedPhysics::DoPhysicsDemo1()
{
	iteration_counter_++;
	// print timings
	static int outCnt;
	if(fc_->printDebuggingInfo==1) outCnt = 1;
	else outCnt = 20;

#ifdef SPH_PROFILING

	++frame_counter_;
	if (frame_counter_ % SPH_PROFILING_FREQ == 0)
		std::cout << "simulation statistics after " << frame_counter_ << " frames: " << std::endl;
	sr::sys::Timer timer;
	timer.Start();

#endif

#ifdef USE_CUDA	

	CudaParticlePhysics();

#else

	CpuParticlePhysics();

#endif	// #ifdef USE_CUDA	

	// select one element and perform a neighborhood query
	// selected_particle = random() % particles.size();
	// GetNeighbors(selected_particle);

#ifdef USE_CUDA	
	//*
	if(fc_->saveParticles == true && iteration_counter_ % outCnt == 0)
	{
		WritePpm(WINDOW_WIDTH, WINDOW_HEIGHT);
		/*
		myUnifiedIO->saveLiquidParticleInfoStaticBoundaryPCISPH(particle_info_for_rendering_.p_pos_zindex, 
																particle_info_for_rendering_.p_vel,
																particle_info_for_rendering_.p_corr_pressure,
																particle_info_for_rendering_.p_predicted_density,
																particle_info_for_rendering_.p_type, 
																fc->numLiquidParticles);
		//*/

		my_unified_io_->SaveLiquidParticleInfoFull(particle_info_for_rendering_.p_pos_zindex, particle_info_for_rendering_.p_type, dptr_.particleCountRounded, fc_->numLiquidParticles);

		my_unified_io_->SaveRigidParticleInfoFull(particle_info_for_rendering_.p_pos_zindex, particle_info_for_rendering_.p_type, dptr_.particleCountRounded, fc_->numRigidParticles);
		
		/*
		myUnifiedIO->SaveLiquidParticleInfoForLoading(particle_info_for_rendering_.p_pos_zindex, particle_info_for_rendering_.p_vel, particle_info_for_rendering_.p_corr_pressure, 
													  particle_info_for_rendering_.p_predicted_density, particle_info_for_rendering_.p_type, dptr_.particleCountRounded, fc->numLiquidParticles);
		//*/											  
		
	}
	//*/
#else
	//*
	if(fc_->saveParticles == true && iteration_counter_ % outCnt == 0)
	{
		WritePpm(WINDOW_WIDTH, WINDOW_HEIGHT);
		my_unified_io_->SaveParticlePositions(particles_);
		fc_->elapsedRealTime += fc_->deltaT;
		std::cout << "tElapsedRealTime: " << fc_->elapsedRealTime << "s, " << fc_->elapsedRealTime / 60.0 << "min" << std::endl;
	}
	//*/
#endif
	
#ifdef SPH_PROFILING

	timer.Stop();
	time_counter_ += timer.GetElapsedTime();
	if (frame_counter_ % SPH_PROFILING_FREQ == 0)
	{
		float averageElapsed = time_counter_ / frame_counter_;
		std::cout << "  overall simulation          average: ";
		std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
		std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
	}

#endif

}

void UnifiedPhysics::DoPhysicsDemo2()
{
	iteration_counter_++;

	const float dtCreateParticle = fc_->particleSpacing/fc_->initialVelocity[0] * 2.0;
	static int layerCnt = 0;
	static int frameCnt = 0;

	if(elapsed_real_time_ >= layerCnt * dtCreateParticle)
	{
		if((dptr_.particleCount + dtCreateParticle) <= dptr_.finalParticleCount )
		{
			//CopyParticleLayerHostToDevice2(dptr_, ppf_particle_count_, ppf_pos_zindex_, ppf_vel_pressure_, ppf_zindex_);
			//CopyParticleLayerHostToDevice(dptr_, ppf_particle_count_);
#ifdef SPH_PROFILING  
			std::cout << "Layer generated at : " << frame_counter_ << std::endl;
#endif
			layerCnt++;
		}
	}

#ifdef USE_CUDA	

	CudaParticlePhysics();

#endif

#ifdef SPH_PROFILING 
	++frame_counter_;
#  endif
	elapsed_real_time_ += fc_->deltaT;

}


//------------------------------------------------------------------------------------------
// end physics
//------------------------------------------------------------------------------------------

void UnifiedPhysics::DrawBox(BBox& bbox)
{
	glLineWidth(1.5);
	glBegin(GL_LINE_LOOP);
	glVertex3f(bbox.getMin()[0], bbox.getMin()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMin()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMax()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMax()[1], bbox.getMin()[2]);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(bbox.getMax()[0], bbox.getMax()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMax()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMin()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMin()[1], bbox.getMax()[2]);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(bbox.getMin()[0], bbox.getMin()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMin()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMin()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMin()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMax()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMax()[1], bbox.getMax()[2]);
	glVertex3f(bbox.getMin()[0], bbox.getMax()[1], bbox.getMin()[2]);
	glVertex3f(bbox.getMax()[0], bbox.getMax()[1], bbox.getMin()[2]);
	glEnd();

}

void UnifiedPhysics::DrawGroundGrid(const BBox& realBoxContainer)
{
	const float height = realBoxContainer.getMin().y;
	const float min_value = -1.2f;
	const float max_value = 2.4f;
	glBegin ( GL_LINES );		
	for (float n = min_value; n <= max_value; n += 0.2 ) {
		glVertex3f ( min_value, height, n );
		glVertex3f ( max_value, height, n );
		glVertex3f ( n, height, min_value );
		glVertex3f ( n, height, max_value );
	}
	glEnd ();
}

void UnifiedPhysics::DrawAsPoints()
{
	int i;
	const float blue[4] = {0.0, 0.3, 0.5, 1.0};
	const float scale_color = 1.0 / ((unsigned int)1 << 29);
	std::vector<UnifiedParticle>::iterator p;

	// setup default material
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

	// set point rendering mode
	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);

	// setup per-point color mode
	glColor4fv(blue);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue);


	// set lighting if enabled
	/*
	if (sr::SphWidget::getLight())
	glEnable(GL_LIGHTING);
	else
	glDisable(GL_LIGHTING);
	*/
	glEnable(GL_LIGHTING);
	// offset for lower-left-bottom corner
	//glTranslatef(virtualBoundingBox.getMin().x, virtualBoundingBox.getMin().y, virtualBoundingBox.getMin().z);

	//WritePpm(500, 500);	
}

void UnifiedPhysics::WritePpm(int width, int height)
{
	//cout << "WritePpm" << endl;

	std::ofstream file_out;
	char filename[1000];
	char filenamejpg[1000];
	static int fileCounter = 0;
	//sprintf(filename, "openglWindow/openGLFrame%04d.ppm", fileCounter);
	sprintf(filename, "openglWindow/openGLFrame%04d.ppm", fileCounter);
	sprintf(filenamejpg, "openglWindow/openGLFrame%04d.jpeg", fileCounter);

	fileCounter++;
	file_out.open(filename, std::ios::binary);

	// read the header
	//char p, nr;
	int maxrgb;

	//cerr << "width = " << width << ", height = " <<  height << endl;
	maxrgb = 255;
	file_out << "P" << "6" << std::endl;
	file_out << width << " " << height << std::endl;
	file_out << maxrgb  << std::endl;

	// allocate memory
	int numBytes = width * height * 3;
	unsigned char *data=0;
	data = new unsigned char[numBytes];

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	// read
	char c;
	int index;

	// go to new line
	//file_in.read(&c, 1);

	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			// read in r, g, b
			for(int rgb = 0; rgb < 3; rgb++)
			{
				index = ((height - 1 - j) * width + i) * 3 + rgb;
				c = (char)data[index];
				file_out.write(&c, 1);
			}
		}
	}

	// close the file
	file_out.close();

	std::string s1(filename);
	std::string s2(filenamejpg);

	// windows空格问题，试试 将 Program Files 改成 progra~1
	// D:/progra~1/
	std::string command = "cjpeg.exe " + std::string(filename) + " > " + std::string(filenamejpg);
	system(command.c_str());
	remove(filename);

	// free memory
	delete[] data;
}
