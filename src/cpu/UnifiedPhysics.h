#ifndef GPU_UNIFIED_PHYSICS_H_
#define GPU_UNIFIED_PHYSICS_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <windows.h>

#include "cudaSPH_Include.h"
#include "UnifiedConstants.h"
#include "vmmlib/vector3.h"
#include "WeightingKernel.h"
#include "UnifiedParticle.h"
#include "UnifiedIO.h"
#include "RigidBody.h"
#include "global.h"

class ParticleRenderer;

class UnifiedPhysics {
public:

	// This struct contains all particle informations for off-line rendering
	struct ParticleInfoForRendering
	{
		float* p_pos_zindex;						
		float* p_vel;
		float* p_corr_pressure;
		float* p_predicted_density;
		int*   p_type;	
		int*   p_activeType;
	};

	UnifiedPhysics(UnifiedConstants* fc);
	~UnifiedPhysics();

	void InitDptr(uint particleCount, float spacing);
	void InitCudaVariables(size_t particleCount, float spacing);	

	void CopyGPUParticles(UnifiedPhysics::ParticleInfoForRendering& particleInfo);

#ifdef DUMP_PARTICLES_TO_FILE
	void DumpGPUParamsToFile();
#endif

	std::vector<UnifiedParticle>& particles() { return particles_; }
	std::vector< std::vector<int> >& neighbor_indices() { return neighbor_indices_; }
	dataPointers& data_pointers() { return dptr_; }

	void SortParticles();

	void UpdateRigidBodyParticleInformationOMP();

	void CreateInitParticles(float spacing, float jitter, BBox& fluidBox); 
	void InitNeighbors(const int numParticles); // initiate neighbors & solidNeighs & solidNeighDistances
	void UpdateNeighbors();	
	void SetParticleCount(uint32_t count);

	void DrawBox(BBox& bbox);
	void DrawGroundGrid(const BBox& realBoxContainer);
	void DrawAsPoints();
	void WritePpm(int width, int height);

	void GetNeighbors(const int i);

	void ComputePhysics(const int i);

	void CalculateDensityPressure(const int i);
	void CalculateDensityPressurePureFluidSPH(const int i);
	float CalculatePressureSPH(const float density);
	float CalculatePressureWCSPH(const float density);
	void CalculateForcesPerParticleStep1(const int i);
	void CalculateForcesPerParticleStep2(const int i);
	vmml::Vector3f CalculateLiquidParticleForces(UnifiedParticle& p, const int i);
	vmml::Vector3f CalculateRigidParticleForces(UnifiedParticle& p, const int i);
	vmml::Vector3f CalculateRigidParticleBoundaryForces(const vmml::Vector3f& pos, const vmml::Vector3f& vel);
	void CalculateForcesWithContactForces(const int i);
	void CalculateForcesWithoutBoundaryControllingForces(const int i);
	void CalculateForcesPerParticleStep3ForElasticBodies(const int i);
	void CalculateRigidBodyForceTorqueMomenta(const int i);
	void UpdateLiquidParticlePosVelSphCPU(const int i);
	void SynRigidParticlesPerRigidBody(RigidBody* body);
	void ApplyRotationToParticlesHost(RigidBody& body);


	/************************************************************************                       
	Simulation Method                           
	************************************************************************/

	void DoPhysics();
	void DoPhysicsDemo1();
	void DoPhysicsDemo2();

	// CPU version
	void CpuParticlePhysics();

	void CpuParticlePhysicsPureFluidsSPH();
	void CpuParticlePhysicsStaticBoundariesSPH();		// simple penalty based method
	void CpuParticlePhysicsPureFluidsPCISPH();
	void CpuParticlePhysicsSimpleRigidFluidCoupling();	
	void CalculateForcePerParticleWithSimpleBoxCollisionHandlingOMP();
	void CalculateExternalForcesPCISPHOMP();
	void CalculateExternalForcesStaticBoundariesPCISPHOMP();
	void CalculateExternalForcesVersatilCouplingPCISPHOMP();
	void UpdateLiquidParticlePosVelCPUOMP();
	void NeighborSearchOMP();
	void CalculateWeightedVolumeOMP();
	void CorrectedDensityPressureComputationOMP();
	void ForceComputationVersatileCouplingOMP();
	void UpdateLiquidParticlePosVelSphOMP();
	void RigidBodyIntegrationOMP();
	void MeltingFreezingOMP();
	void CalculateLiquidParticleDensityPressureOMP();
	void CalculateRigidBodyMomentaOMP();
	void SynRigidParticlesOMP();
	void UpdateLiquidParticlePosVelSphCPUOMP();

	void CpuParticlePhysicsBoundaryForceMonaghan();				// simple boundary force method -> Monaghan 2005 "Smoothed particle hydrodynamics" P1749
	void CpuParticlePhysicsWithOneWaySolidToFluidCoupling();	// simple direct forcing method   TODO: it's not complemented yet.
	void CpuParticlePhysicsVersatileCouplingPCISPH();				// "Versatile Rigid-Fluid Coupling for Incompressible SPH" -> WCSPH/PCISPH

	// GPU version
	void CudaParticlePhysics();

	// surface tension & cohesion
	vmml::Vector3f CalculateSurfaceTensionForcePCISPH(const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, const float splineCoefficient);
	vmml::Vector3f CalculateSurfaceCohesionForcePCISPH(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, float neigh_weighted_vol, WeightingKernel& weightingKernel);

	/************************************************************************
	Boundary handling	                   
	************************************************************************/

	void BoundaryHandlingBoxSimple(const int i);
	void BoundaryHandlingBoxOMP();
	void BoundaryHandlingBoxPerParticleOMP();
	void BoundaryHandlingPureFluidOMP();
	void BoundaryHandlingBoxPerNonRigidParticle(const vmml::Vector3f& minBox, const vmml::Vector3f& maxBox, const float damping, vmml::Vector3f& pos, vmml::Vector3f& vel);

	// ---------------------- boundary force gamma Monaghan 2005 "Smoothed particle hydrodynamics" P1749 ----------------------
	float BoundaryForceGamma(float normalDistance, float sr);
	float BoundaryForceChi(float tangentialDistance, float boundaryParticleSpacing);
	float BoundaryForceBxy(float tangentialDistance, float normalDistance, float sr, float boundaryParticleSpacing);
	void CalculateBoundaryForceMonaghan(const int i);
	//---------------------------------------------------------------------------------------------------------------

	void CpuParticlePhysicsStaticBoundariesPCISPH();
	vmml::Vector3f CalculateBoundaryFluidPressureForceHost(uint dist_lut, vmml::Vector3f p_pos, vmml::Vector3f neigh_pos, float p_density, float p_corr_pressure, float weighted_vol);

	vmml::Vector3f BoxBoundaryForce(const int i);
	void CollisionHandling(vmml::Vector3f &pos, vmml::Vector3f &vel);
	void PredictIntegrationOneWaySolidToFluidCoupling(const int i);
	void DetectCorrectFluidRigidCollision(const int i);

	// ---------------------- wall weight function method from "Smoothed particle hydrodynamics on gpus" ----------------------
	// pre-compute the distance from each particle to all the planar boundaries & select the minimum distance
	void PrecomputeDistanceFunction();		// lookup table in cpu version & 3D texture in gpu version
	void PrecomputeWallWeightFunction();	// lookup table in cpu version & 1D texture in gpu version
	void TestWallWeightFunction();
	float DistanceToWall(const float& x, const float& y, const float& z, vmml::Vector3f& norm = vmml::Vector3f(0.0f, 0.0f, 0.0f));
	float WallWeight(float distToWall, float effectiveDist);
	float PythagoreanDist(float x, float y){ return sqrt(x*x + y*y); }	// z^2 = x^2 + y^2
	float GetDistToWallLut(float x, float y, float z);
	float GetWallWeightFunctionLut(float distToWall);
	vmml::Vector3f GetBoundaryPressureForceWallWeight(const int i);

	// --------------------------------------- versatile coupling ---------------------------------------------------
	void CalculateWeightedVolumeWithshepard_filter_den_kernel_factor(const int i);						// use together with CalculateCorrectedDensitiesVersatileCouplingWithshepard_filter_den_kernel_factor();
	void CalculateCorrectedDensitiesVersatileCouplingWithshepard_filter_den_kernel_factor(const int i);	// incorrect
	// versatile method for pcisph
	void PredictionCorrectionStepVersatileCoupling();
	void CalculateWeightedVolume(const int i);	
	void CalculateCorrectedDensitiesPressureVersatileCoupling(const int i);
	void CalculateParticleForceVersatilCoupling(const int i);
	void CalculateForcesISPHVersatilCoupling(const int i);
	void ComputeCorrectivePressureForceVersatilCoupling(const int i);
	//---------------------------------------------------------------------------------------------------------------

	/************************************************************************
	PCISPH methods	  	                    
	************************************************************************/

	void CalculateExternalForcesPCISPH(const int i);
	void CalculateExternalForcesStaticBoundariesPCISPH(const int i);
	void CalculateExternalForcesVersatilCouplingPCISPH(const int i);
	void PredictionCorrectionStep();	
	void PredictPositionAndVelocity(const int i);
	void ComputePredictedDensityAndPressure(const int i);
	void ComputeCorrectivePressureForce(const int i);
	void ComputeDensityErrorFactor();
	void ComputeGradWValues();
	void CreatePreParticlesISPH();
	void ComputeFactor();

#ifdef SPH_DEMO_SCENE_2
	void CreateParticleLayer();
#endif

	/**
	* Freezes the particles from startIndex to endIndex so they can become a solid object.
	* This should not be called during a simulation, but beforehand.
	*/
	void SolidifyRange(const int startIndex, const int endIndex);

	/**
	* freezes a particle so it can become a solid object
	* This should not be called during a simulation, but beforehand.
	*/
	void SolidifyParticle(const int i);

	/**
	* Add particles using this method so UnifiedPhysics can initialize them.
	*/
	void Add(UnifiedParticle* newParticle);

	/************************************************************************
	Rigid body properties                       
	************************************************************************/

	/**
	* Adds particles[pIndex] to this (and sets its parent to this) and adapts
	* angularMomentum, velocity, inertia, mass, centerOfMass, init_relative_pos. Inertia must 
	* still be inverted; neither torque nor force are changed.
	*
	* Velocities of part and the other particles are not changed -> should this be done?
	*/
	void AddToRigidBody(RigidBody* rigid, const int pIndex);

	/**
	* Adds the particles of part to this (and sets their parent to this) 
	* and adapts angularMomentum, velocity, inertia, mass, 
	* centerOfMass, init_relative_pos. Inertia must still be inverted; neither torque nor 
	* force are changed. part must still be deleted.
	*
	* Velocities of the particles are not changed -> should this be done?
	*/
	void AddRigidBodyParticles(RigidBody* currentRigid, RigidBody* part);

	/**
	* Removes part from this by adapting
	* angularMomentum, velocity, inertia, mass, centerOfMass, but part is
	* still in the vector and has this as its parent. Inertia must 
	* still be inverted; neither torque nor force are changed.
	*
	* Velocities of the remaining particles are not changed
	*/
	void RemoveFromRigidBody(RigidBody* currentRigid, UnifiedParticle* part);

	/**
	* Removes part from this and adapts angularMomentum, velocity, 
	* inertia, mass, centerOfMass, but does not remove the individual 
	* particles. Inertia must still be inverted; neither torque nor 
	* force are changed.
	*
	* Velocities of the remaining particles are not changed.
	*/
	void RemoveRigidBodyParticles(RigidBody* currentRigid, RigidBody* part);

	/**
	* change the state hasMadeSolid of particles from startIndex to endIndex
	*/
	void MarkMadeSolid(const int startIndex, const int endIndex);

	/**
	* Updates the neighborhoods of the particles between begin and end, taking
	* only these particles into account.
	*/
	void UpdateNeighborsRangeParallel(const int startIndex, const int endIndex);

	/**
	* Compute local inertial tensor for rigid body composed of equally weighted particles
	*/
	Matrix3x3 ComputeLocalInertiaTensorRigidBody(const RigidBody& body);

	/**
	* return the center of mass of a rigid body composed of equally weighted particles
	*/
	vmml::Vector3f ComputeCenterOfMass(std::vector<UnifiedParticle>& particles_);

	vmml::Vector3f& GetCenterOfMassStaticBoundaryBox() { return center_of_mass_static_boundary_box_; }
	const vmml::Vector3f& GetCenterOfMassStaticBoundaryBox() const { return center_of_mass_static_boundary_box_; }

	Matrix3x3& GetInertiaTensorStaticBoundaryBox() { return inertia_tensor_static_boundary_box; }
	const Matrix3x3& GetInertiaTensorStaticBoundaryBox() const { return inertia_tensor_static_boundary_box; }
	void ColorRamp(float t, float *r);			


	/************************************************************************
	OpenGL VBO 				                
	************************************************************************/

	void GlVBOInit();
	void GlVBOInitWithoutFrozenParticles();
	void UpdatePositionVBO();
	void UpdatePositionVBOWithoutFrozenParticles();
	unsigned int GetVBOCpu() const
	{
		return m_pos_vbo_cpu_;
	}

	size_t GetNumNonfrozenParticles() const{ return num_nonfrozen_particles_; }

	/************************************************************************
	CUDA/Graphics Interoperability 	               
	************************************************************************/

	uint CreateVBO(uint size);

	void DoCudaGLInteraction();

	unsigned int GetCurrentReadBuffer() const
	{
		return dptr_.m_posVbo[m_current_read_];
	}

	unsigned int GetColorBuffer() const
	{
		return m_color_vbo_;
	}

	ParticleInfoForRendering& particle_info_for_rendering() { return particle_info_for_rendering_; }
	ParticleRenderer* renderer() { return renderer_; }
	UnifiedIO* my_unified_io() { return my_unified_io_; }

private:

	//----------------------- TODO: delete ---------------------------------------------------------
	// these functions should be placed in WeightingKernel class
	void CalculateKernelConstants();	
	float KernelDensity(float distSq);		
	float KernelPressureGrad(float dist);	
	float KernelViscosityLaplacian(float dist);
	void KernelSurfaceTension(float distSq, float& grad, float &lap);
	float KernelElasticGrad(float dist);		// Gradient kernel used for the elastic forces ("sine kernel")
	//----------------------------------------------------------------------------------------------

	void HeatingFluid(const int i);

	/**
	* For MLS: calculates INV(sum_j(x_ij * x_ij [transposed])*w_ij), 
	* where x_ij is the vector between i & j and w_ij is the kernel
	* the moment matrix in Eq.(14) from "Point Based Animation of Elastic, Plastic and Melting Objects"
	*/
	void CalculateInvertedMomentMatrix(const int pindex);

	/**
	* Updates the plastic strain of p based on the measured elasticStrain
	*/
	void UpdateStrain(UnifiedParticle* p, float* elasticStrain);

	/**
	* Updates the state of aggregation of all particles (freezes, melts, and 
	* adapts properties depending on temperature)
	*/
	void UpdateStateOfAggregation();

	/**
	* Connects all neighs of currentParticle that are cold enough to it
	* (regardless of the temperature of currentParticle)
	*/
	void FreezeParticle(const int currentParticleIndex);

	/**
	* Has the particle forget about his solidNeighs (regardless of the 
	* temperature of currentParticle)
	*/
	void MeltParticle(const int pIndex);

	/**
	* Removes linkingParticle from its object and splits the object if 
	* linkingParticle was a critical link in the object.
	*/
	void SplitRigidBody(const int linkingParticleIndex);

	/**
	* Helper method for SplitRigidBody(const int linkingParticleIndex): recurses over the 
	* neighbors of currentParticle and adds them to collectedParticles if 
	* they should belong to the same RigidBody as currentParticle.
	*/
	void SplitRecursively(const int currentParticleIndex, std::vector<int>& collectedParticles);

	vmml::Vector3f CalculateSurfaceTensionForcePCISPHHost(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos);
	vmml::Vector3f CalculateSurfaceCohesionForcePCISPHHost(const uint dist_lut, const float dist, const vmml::Vector3f p_pos, const vmml::Vector3f neigh_pos, float neigh_weighted_vol);

	// Rigid body dynamics
	vmml::Vector3f CalculateSpringForceHost(const float& dist, const float& overlap, const vmml::Vector3f& r_ij);
	vmml::Vector3f CalculateDampingForceHost(const vmml::Vector3f& v_ij);
	vmml::Vector3f CalculateBoundaryForcePerLiquidParticleHost(const vmml::Vector3f& position);
	vmml::Vector3f CalculateBoundaryForcePerRigidParticleHost(const vmml::Vector3f& position, const vmml::Vector3f& vel);

	void CreatePartilcesCPU(float spacing, float jitter, BBox& fluidBox);

	void AddPovInputFileNames(std::vector<std::string> &input_file_names); 

	inline float Lerp(float a, float b, float t)
	{
		return a + t*(b-a);
	}

private:

	std::vector<UnifiedParticle> particles_;								// stores actual particles
	std::vector< std::vector<int> > neighbor_indices_;							// stores neighbor particle indices (in the same order)
	std::vector<RigidBody*> rigid_bodies_;								// Contains all rigid bodies currently in the simulation
	std::vector< std::vector<int> > solid_neighs_;						// Contains neighbors that belong to this particle's body (in the same order)
	std::vector< std::vector<vmml::Vector3f> > solid_neigh_distances_;		// Reference distance vectors to the solidNeighs (in the same order)

	/************************************************************************/
	/*                CUDA/Graphics Interoperability : BEGIN                */
	/************************************************************************/

	// CPU data
	float *h_positions_;											// particle positions
	float *h_pos_without_frozen_;									// non frozen particle positions

	uint   m_pos_vbo_cpu_;											// vertex buffer object for CPU version
	uint   m_color_vbo_;											// vertex buffer object for colors

	struct cudaGraphicsResource *cuda_posvbo_resource_[2];		// handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *cuda_colorvbo_resource_;		// handles OpenGL-CUDA exchange

	ParticleRenderer* renderer_;

	uint m_current_read_;
	uint m_current_write_;

	/************************************************************************/
	/*                CUDA/Graphics Interoperability : END                  */
	/************************************************************************/
	ParticleInfoForRendering particle_info_for_rendering_;

	int iteration_counter_;

	UnifiedConstants *fc_;
	UnifiedIO* my_unified_io_;
	WeightingKernel *my_kernel_;
	float kernel_self_;

	float box_width_;
	float box_length_;
	float box_height_;

	//----------------------- TODO: delete ---------------------------------
	// these variables should be placed in WeightingKernel class
	float kernel_density_const_;
	float kernel_pressure_const_;
	float kernel_viscosity_const_;
	float kernel_surface_tension_const_;
	float kernel_elastic_const_;									// Constant part of the sine kernel computation
	//----------------------------------------------------------------------

	float clap_self_;
	float support_radius_sq_;

	// PCISPH params
	bool density_error_too_large_;
	int max_loops_sim_;
	float avg_loops_sim_;
	float max_predicted_density_;

	// debugging/stats
	float min_dens_;	
	float max_dens_;
	float dens_avg_timestep_;

#ifdef SPH_PROFILING
	unsigned int frame_counter_;
	float        time_counter_;
	float        time_counter_rendering_;
	float        time_counter_total_;
#endif

	//#ifdef SPH_DEMO_SCENE_2
	float elapsed_real_time_;
	uint32_t particle_count_;
	uint32_t ppf_particle_count_;
	float* ppf_pos_zindex_;
	float* ppf_vel_pressure_;
	unsigned int* ppf_zindex_;
	//#endif 

	uint* indices_g_;
	dataPointers dptr_;

	FILE* ffmpeg_;

	int* buffer_;

	size_t num_nonfrozen_particles_;	// non frozen particle number

	/************************************************************************
	Rigid body properties                       
	************************************************************************/
	// inertia tensor of particle based rigid bodies
	vmml::Vector3f center_of_mass_static_boundary_box_;
	Matrix3x3 inertia_tensor_static_boundary_box;

	// lookup table for wall weight function & distance function
	static int const lut_size_distance_ = 300;		// refer to WeightingKernel.h	// TODO: choose an appropriate value for cuda program
	static int const lut_size_wall_weight_ = 300;
	float wall_weight_function_[lut_size_wall_weight_];
	float distance_function_[lut_size_distance_][lut_size_distance_][lut_size_distance_];

};

#endif	// GPU_UNIFIED_PHYSICS_H_
