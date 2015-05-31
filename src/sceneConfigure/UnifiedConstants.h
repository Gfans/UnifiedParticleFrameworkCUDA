#ifndef UNIFIEDCONSTANTS__H
#define UNIFIEDCONSTANTS__H

#include "vmmlib/vector3.h"
#include "vmmlib/axisAlignedBoundingBox.h"
#include "global.h"

//#define SPH_DEMO_SCENE_2
//#define DUMP_PARTICLES_TO_FILE

typedef vmml::AxisAlignedBoundingBox<float> BBox;

class UnifiedConstants
{
public:

	UnifiedConstants();
	~UnifiedConstants();

	void setParticleSpacing(float spacing);

private:

	void setSimulationMethod(int method);

	void computeGasConstAndTimeStep(float densityVariation);

	void setBoundingBoxValues();

	void setBBox(const vmml::Vector3f& minBoxVec, const vmml::Vector3f& maxBoxVec, BBox& box)			{ box.set(minBoxVec, maxBoxVec); } 

public:

	// animation loop or individual steps can be controlled by key inputs
	int animating;
	int animationStep; 
	int simulationMethod;			// 0: SPH 1: WCSPH  2: PCISPH

	float sizeFactor;				// defines particle resolution

	float fluidLength;
	float fluidWidth;
	float fluidHeight;

	float rigidLength;
	float rigidWidth;
	float rigidHeight;

	float cube_side_length;

	// virtual z-index bounding box
	float virtualIndexingBoxLength;
	float virtualIndexingBoxWidth;
	float virtualIndexingBoxHeight;

	// bounding box
	float boxLength;
	float boxWidth;
	float boxHeight;

	// elastic sheet
	float sheetLength;
	float sheetWidth;

	bool readParticles;				// read particles from file
	bool saveParticles;				// write particle positions into file

	// physical quantities
	float gravityConst;	
	float fluidRestDensity;
	float fluidRestVol;				// for liquid particles, we assume it to be constant for WCSPH & PCISPH, i.e. initialMass / fluidRestDensity;
	float relativeFluidJitter;
	float globalSupportRadius;
	float particleSpacing;
	float particleRenderingSize;	// for vbo rendering
	float particleRadius;			// for collision detection & reaction in rigid body & granular simulation 
	float collisionDistThreshold;	// also for collision detection & reaction
	float initialMass;
	float minSurfaceGrad;
	float fluidGasConst;
	float fluidGasConstantWCSPH;
	float fluidViscConst;	
	//float viscFactor;
	float distToCenterMassCutoff;
	float gradientVelCutoff;

	bool useOldRigidBodyMethod;		// debugging purpose for two different rigid body methods, TODO: delete after fixing bugs in the second method

	// Versatile coupling
	float alpha;					// viscosity constant between liquid particles in "Versatile Rigid-Fluid Coupling for Incompressible SPH"	TODO: combined with fluidViscConst
	float rho;						// viscosity constant between rigid & liquid particles
	float epsilon;					// E.q(11) from "Versatile Rigid-Fluid Coupling for Incompressible SPH"	TODO: combined with fluidViscConst
	float epsilon_h_square;			// epsilon * h * h
	float nu_liquid;				// viscous factor between fluid particles in E.q(11)
	float nu_rigid_liquid;			// viscous factor between fluid & rigid particles in E.q(14)

	float gamma;

	// Contact forces
	float k_r;						// elastic restoration coefficient, this parameter controls the particle stiffness	
	float k_d;						// viscous damping coefficient, this parameter controls the dissipation during collisions
	float k_t; 						// Particle-Based Simulation of granular Materials (16) viscous damping term
	float mu; 						// particle friction Coefficient between each other 

	// time step
	float deltaT; 
	float deltaT_sph;
	float deltaT_wcsph;
	float deltaT_pcisph;
	float deltaT_RigidBody;

	// concentration and temperature diffusion
	float diffusionConcConst;
	float diffusionTempConst;

	/************************************************************************/
	/*                             Boundary handling	                    */
	/************************************************************************/
	// "Direct forcing for Lagrangian rigid-fluid coupling"
	float coefficient_restitution;  // [0.0, 1.0] 1.0 : perfectly elastic collision 0.0 : perfectly inelastic collision
	float coefficient_slip;			// [0.0, 1.0] it can be used to damp the relative tangential velocity of the fluid and the rigid body

	bool useVersatilCoupling;		// versatile method

	// "Real-Time Rigid Body Interactions"
	float springCoefficient;
	float dampingCoefficient;
	float shearCoefficient;
	float springCoefficientBoundary;

	//Z-index related bounding box
	BBox virtualBoundingBox;
	BBox realBoxContainer;
	BBox collisionBox;				// penalty force related
	BBox fluidBox;
	BBox rigidBox;
	BBox rigidCube;
	vmml::Vector3f zindexStartingVec;
	vmml::Vector3f scales;				// number of virtual grid cells(used for z-indexing) per unit length in each direction	

	// visualization
	bool drawTemperature;
	bool drawRigidParticle;
	bool drawSoftParticle;
	bool drawLiquidParticle;
	bool drawGranularParticle;
	bool drawClothParticle;
	bool drawSmokeParticle;
	bool drawFrozenParticle;
	bool drawVelocity;
	bool drawPressureForce;

	bool heatingGround;				// heating
	float automaticCooling;			// particles are cooled every time step by this amount

	float wallX;

	float pipeRadius;
	float pipeHeight;
	vmml::Vector3<float> pipePoint1;
	vmml::Vector3<float> pipePoint2;

	float initialHeight;
	vmml::Vector3<float> initialVelocity;

	float fluidViscConst_tube;		// for water in tube 
	//float viscFactor_tube;

	// visualization
	bool drawNothing;
	bool displayEnabled;			
	int drawParticles;

	// used for time step determination according to courant condition
	float courantFactor;
	float relevantSpeed;
	float speedOfSound;

	// boundary force values
	bool addBoundaryForce;
	float maxBoundaryForce;
	float boundaryForceFactor;
	float forceDistance;			// forceDistant = particle radius r, particle rest spacing = 2*r, global support radius = 4*r
	float distance_frozen_to_boundary;
	bool addWallWeightFunction; 

	// use Ihmsen's boundary handling method
	bool useIhmsen2010Method;

	bool useSurfaceTension;
	bool isTwoWayCoupling;			// disable this to get a faster simulation for pure fluid simulation

	float surface_tension_gamma;
	float surface_adhesion_beta;

	float elapsedRealTime;			// measures the elapsed "real" time

	// defines the simulation type
	// o: original SPH
	// w: WCSPH
	// i: incompressible SPH (PCISPH)
	char physicsType;

	SpecificSimAlgorithm specificSimAlgorithm;

	// PCISPH params
	int maxLoops;
	int minLoops;
	float densityErrorFactor;
	float densityErrorFactorParameter; 
	float maxDensityErrorAllowed;
	int printDebuggingInfo;
	int numLiquidParticles;
	int numRigidParticles;

	// rigid body variables
	float terminalSpeed;

};

#endif	// UNIFIEDCONSTANTS__H




