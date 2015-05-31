#ifndef SCENE__H
#define SCENE__H SCENE__H

#include "UnifiedPhysics.h"
#include "UnifiedConstants.h"
#include "UnifiedParticle.h"
#include "vmmlib/vector3.h"
#include "ObjectLoader.h"

/**
* The Scene class is used to create scenes that will then be simulated. It 
* provides methods to load 3D models, load particle positions from files etc. 
* The build(UnifiedPhysics*, UnifiedConstants*) method can be changed in order to
* create a new scene setup.
*/
class Scene
{
public:
	Scene();
	~Scene();
	/**
	* Adds particles to the particle vector of physics. Note that any 
	* pointer to a particle inside physics->particle will be invalid if any
	* particle is added here!
	*/
	void build(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, int& num_frozen_particles_);

protected:
	/**
	* Holds some particles that can be made solid, moved etc.
	*/
	class ParticleObject;
	/**
	* add a ParticleObject to the scene
	*/
	void add(ParticleObject& po);
	/**
	* Loads a 3D model file. If scaling and particleTemplate are the same,
	* loading the model once and copying the ParticleObject is more 
	* efficient than reloading.
	* loader - subclass of ObjectLoader that knows how to load 'fileName'
	* fileName - the file to load
	* scaleX, scaleY, scaleZ - scaling of the object
	* spacing - the loader uses this to calculate the difference between particles
	* particleTemplate - particle used for the object
	*
	*/
	ParticleObject LoadModel(ObjectLoader* loader, const std::string& file_name, const float scale_x, const float scale_y, const float scale_z, const float spacing, UnifiedParticle& particle_template);
	/**
	* Reads the particles saved in file.
	*/
	ParticleObject readInParticles(const std::string& file, UnifiedConstants* fc);
	/**
	* Saves the particles of physics in a file called
	* '[firstPartOfFileName][fileCounter].txt'.
	* THIS IS CURRENTLY STILL DONE BY UnifiedPhysics.
	*/
	void saveParticles(UnifiedPhysics* physics);
	/**
	* Creates frozen particles at the boundary of the simulation
	* spacing - the distance between the frozen particles.
	*/
	ParticleObject createBoxBoundaryFrozenParticles(UnifiedPhysics* physics, float spacing, UnifiedConstants* constants, float jitter, const vmml::Vector3f &offset);
	/**
	* Creates frozen particles at the boundary of the simulation in Multi Layers
	* spacing - the distance between the frozen particles.
	*/
	ParticleObject createBoxBoundaryFrozenParticlesMultiLayers(UnifiedPhysics* physics, float spacing, UnifiedConstants* constants, float jitter, const vmml::Vector3f &offset);
	/**
	* Creates frozen particles at the floor.
	* spacing - the distance between the frozen particles
	*/
	ParticleObject createFrozenParticlesAtBottom(float spacing, UnifiedConstants* constants);
	/**
	* Does the actual adding of the particles to the physics object. Should 
	* be called once, at the end of 
	* build(UnifiedPhysics* physics, UnifiedConstants* constants). 
	*/
	void addAllTo(UnifiedPhysics* physics);

	class ParticleObject 
	{
	public:
		ParticleObject();
		/**
		* Initializes the object with a vector that will be deleted 
		* when it isn't needed anymore.
		*/
		ParticleObject(std::vector<UnifiedParticle>* partVector);
		~ParticleObject();
		/**
		* Copy constructor makes a deep copy of the particle vector.
		*/
		ParticleObject(const ParticleObject& rhs);
		/**
		* Assignment operator makes a deep copy of the particle vector.
		*/
		ParticleObject& operator=(const ParticleObject& rhs);
		/**
		* Adds a particle to the object.
		*/
		void add(const UnifiedParticle& p);
		/**
		* Used to specify if the particles are to be solid.
		*/				 
		void setSolid(bool solid);
		/**
		* Returns true if the particles are set solid.
		*/
		bool isSolid();
		/**
		* Moves the whole object by distance.
		*/
		void move(const vmml::Vector3f& distance);
		/**
		* Gets the position of the object.
		*/
		vmml::Vector3f getPosition();
		/**
		* Returns a reference to the particles in this object
		*/
		std::vector<UnifiedParticle>& getParticles();

	private:
		/**
		* Contains the particles
		*/
		std::vector<UnifiedParticle>* particles;
		/**
		* Specify if the particles are to be solid.
		*/
		bool solid;
		/**
		* Position to be added to the particle positions when the scene
		* is created.
		*/
		vmml::Vector3f positionOffset;
	};

private:
	/**
	* The ParticleObjects that have been added after the last call to 
	* addAllTo(UnifiedPhysics* physics). Use add(ParticleObject& po)
	* to add new ones!
	*/
	std::vector<ParticleObject> particleObjects;
	/**
	* Sum of all particles in the added ParticleObjects.
	*/
	int particleCount;
	/**
	* First part of file name for loading and saving.
	*/
	std::string firstPartOfFileName;

	/**
	* These two are not actually part of the Scene interface. They are just
	* (adapted) copies of the methods in the UnifiedPhysics class with the
	* same name.
	*/
	ParticleObject createParticles(UnifiedConstants* constants);
	ParticleObject createParticlesPeriodicBoundary(UnifiedConstants* fc);

	ParticleObject creatElasticSheet(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter);
	ParticleObject createChromeSphere(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset);
	ParticleObject createFluidObject(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset);
	ParticleObject readFluidObjectFromFile(const char* fileName, UnifiedPhysics* physics, UnifiedConstants* constants, int& numParticles);
	ParticleObject createRigidObject(const BBox& rigidBox, UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset);
	ParticleObject createSurfaceParticlesObject(const std::string filename, const vmml::Vector3f scale, UnifiedPhysics* physics, UnifiedConstants* constants, 
		float spacing, float jitter, const vmml::Vector3f &offset, float density, float rb_density = 1000.0f);

	void CreateFullParticleObject(UnifiedConstants* fc, const vmml::Vector3f &offset, const float density, const int particleType, const char* model_file_name, 
		const vmml::Vector3f &scale, const float particle_spacing, bool isRigidBody);

};

#endif
