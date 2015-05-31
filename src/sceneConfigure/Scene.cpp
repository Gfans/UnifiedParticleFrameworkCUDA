#include "Scene.h"
#include "ObjLoader.h"
#include "global.h"

#include "vector3d.h"
#include "particleplacer.h"

//--------------------------------------------------------------------
Scene::Scene()
	//--------------------------------------------------------------------
{
	firstPartOfFileName = "output/particles";
	particleCount = 0;
}

//--------------------------------------------------------------------
Scene::~Scene()
	//--------------------------------------------------------------------
{
}

//--------------------------------------------------------------------
void Scene::build(UnifiedPhysics* physics, UnifiedConstants* fc, float spacing, float jitter, int& num_frozen_particles_)
	//--------------------------------------------------------------------
{
	// --- Add scene loading code here: ---
	// (or overwrite this method in a subclass, but don't forget to call addAllTo(physics) in the end)

	float my_block_size = fc->scales.x * fc->globalSupportRadius;
	my_block_size = MAX(my_block_size, fc->scales.y * fc->globalSupportRadius); 
	my_block_size = MAX(my_block_size, fc->scales.z * fc->globalSupportRadius); 

	// setup for spatial queries
	zIndex.SetBlockSize(my_block_size);

	vmml::Vector3f offset(0.0f, 0.0f, 0.0f);
	vmml::Vector3f scale(0.0f, 0.0f, 0.0f);

	{
		//* liquid 1
		offset.set(fc->globalSupportRadius + fc->particleSpacing, fc->globalSupportRadius + fc->particleRadius, fc->globalSupportRadius + fc->particleSpacing);	// set as such to make sure liquid particles are set tangent to boundary box
		ParticleObject water1 = createFluidObject(physics, fc, spacing, jitter, offset);
		std::cout << "nofParticles in fluid object: " << water1.getParticles().size() << std::endl;
		fc->numLiquidParticles += water1.getParticles().size();
		for(std::vector<UnifiedParticle>::iterator p=water1.getParticles().begin(); p != water1.getParticles().end(); ++p)
		{
			// Use this water viscosity for all demos
			p->fluid_properties_.visc_const = 1.5f; 
			p->solid_properties_.visc_const = 1.5f; 
			p->type_ = LIQUID_PARTICLE;

			// Use this for the melting bunny
			//p->temperature = 100.0f; // melting !sinking! bunny in hot water
		}
		add(water1);
		//*/
	}

	// rigid bodies
#if 1

	const float rb1_density = 200.0f;
	const float rb2_density = 500.0f;
	const float rb3_density = 2000.0f;
	const float rb4_density = 300.0f;
	const float rb5_density = 700.0f;
	const float rb6_density = 1500.0f;

	bool isRigidBody;

	scalar density = 1.0f;			// 1.2  1.0 -> small gaps between particles   1.5 -> very few gaps (if any)

	jitter = 0.0f;

	//bunnies
	//*bunny 1 full particles object
	{
		const vmml::Vector3f offset(1.0, -0.02, 0.05);
		const vmml::Vector3f scale(2.5, 2.5, 2.5);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb1_density, RIGID_PARTICLE, "models/bunny-1500.obj", scale, spacing, isRigidBody);
	}	
	//*/ 

	//*bunny 2 full particles object
	{
		const vmml::Vector3f offset(1.0, -0.02, 0.45);
		const vmml::Vector3f scale(2.5, 2.5, 2.5);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb2_density, RIGID_PARTICLE, "models/bunny-1500.obj", scale, spacing, isRigidBody);
	}	
	//*/ 

	//*bunny 3 full particles object
	{
		const vmml::Vector3f offset(1.0, -0.02, 0.85);
		const vmml::Vector3f scale(2.5, 2.5, 2.5);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb3_density, RIGID_PARTICLE, "models/bunny-1500.obj", scale, spacing, isRigidBody);
	}	
	//*/ 

	//teapots
	//*teapot 1 full particles object
	{
		const vmml::Vector3f offset(1.4, 0.055, 0.05);
		const vmml::Vector3f scale(0.00025f, 0.00025f, 0.00025f);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb4_density, RIGID_PARTICLE, "models/teapot.obj", scale, spacing, isRigidBody);
	}	
	//*/

	//*teapot 2 full particles object
	{
		const vmml::Vector3f offset(1.4, 0.055, 0.45);
		const vmml::Vector3f scale(0.00025f, 0.00025f, 0.00025f);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb5_density, RIGID_PARTICLE, "models/teapot.obj", scale, spacing, isRigidBody);
	}	
	//*/ 

	//*teapot 3 full particles object
	{
		const vmml::Vector3f offset(1.4, 0.055, 0.85);
		const vmml::Vector3f scale(0.00025f, 0.00025f, 0.00025f);
		isRigidBody = true;
		CreateFullParticleObject(fc, offset, rb6_density, RIGID_PARTICLE, "models/teapot.obj", scale, spacing, isRigidBody);
	}	
	//*/

#endif

	// ------------------------------------
	addAllTo(physics);

	std::cout << "Total Particles in physics world: " << physics->particles().size() << std::endl;

}

//--------------------------------------------------------------------
void Scene::add(ParticleObject& po)
	//--------------------------------------------------------------------
{
	particleObjects.push_back(po);
	particleCount += po.getParticles().size();
}

//--------------------------------------------------------------------
void Scene::addAllTo(UnifiedPhysics* physics)
	//--------------------------------------------------------------------
{
	// make sure there's enough space in the vector

	const unsigned int totalNumParticles = physics->particles().size()+particleCount;

	physics->particles().reserve(totalNumParticles);
	// we also add dummy values for neighbors & solidNeighs & solidNeighDistances
	// we have to make sure neighbors & solidNeighs & solidNeighDistances have the same size as particles vector has
	physics->InitNeighbors(totalNumParticles);
	/*
	physics->neighbors.reserve(totalNumParticles); // do not forget this vector
	physics->solidNeighs.reserve(totalNumParticles); // do not forget this vector
	physics->solidNeighDistances.reserve(totalNumParticles); // do not forget this vector
	*/

	// add all objects
	for(std::vector<ParticleObject>::iterator partObj=particleObjects.begin(); partObj != particleObjects.end(); ++partObj)
	{
		ParticleObject* currentObject = &(*partObj);

		int startOffset = physics->particles().size();

		// move and add the particles
		std::vector<UnifiedParticle>::iterator endIter = currentObject->getParticles().end();
		for(std::vector<UnifiedParticle>::iterator p = currentObject->getParticles().begin(); p != endIter; ++p)
		{
			UnifiedParticle* currentParticle = &(*p);

			currentParticle->position_ += currentObject->getPosition();
			physics->Add(currentParticle);
		}

		// mark the particles hasMadeSolid
		if(currentObject->isSolid())
		{
			physics->MarkMadeSolid(startOffset, physics->particles().size());
		}
	}	

	// we have to guarantee the particles vector be sorted for lower_bound algorithm
	physics->SortParticles();

	physics->UpdateNeighborsRangeParallel(0, physics->particles().size());

	// iterate over all particles
	// make the particles solid if necessary
	const unsigned int num_particles = physics->particles().size();
	for (int i = 0; i < num_particles; ++i)
	{
		if (physics->particles()[i].has_made_solid_)
		{
			physics->SolidifyParticle(i);
		}
	}

	particleObjects.clear();
	particleCount = 0;
}

Scene::ParticleObject Scene::LoadModel(ObjectLoader* loader, const std::string& file_name, const float scale_x, const float scale_y, const float scale_z, const float spacing, UnifiedParticle& particle_template)
{
	std::cout << "Loading 3D model" << std::endl;
	std::vector<UnifiedParticle>* particles = loader->LoadParticles(file_name, scale_x, scale_y, scale_z, spacing, particle_template);
	return ParticleObject(particles);
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::readInParticles(const std::string& file, UnifiedConstants* fc)
{
	std::cout << "read in particles" << std::endl;
	std::ifstream infileParticles;
	std::vector<UnifiedParticle>* particles = new std::vector<UnifiedParticle>();
	float px, py, pz;
	float dens, mass; 
	int type;

	try 
	{
		infileParticles.open(file.c_str());
	}
	catch(...)
	{
		std::cout << "Error opening file" << std::endl;
	}
	while(infileParticles)
	{
		infileParticles >> px >> py >> pz >> dens >> mass >> type;
		if(type!=3)
		{
			particles->push_back(UnifiedParticle(px, py, pz, dens, mass, type));

		}
	}
	return ParticleObject(particles);
}

//--------------------------------------------------------------------
void Scene::saveParticles(UnifiedPhysics* physics)
	//--------------------------------------------------------------------
{
	static int fileCounter = 0;
	char filename[200];
	sprintf(filename, (firstPartOfFileName+"%04d.txt").c_str(), fileCounter);
	std::cout << filename << std::endl;
	std::ofstream outfileParticles;
	outfileParticles.open(filename);
	outfileParticles.clear();

	for (std::vector<UnifiedParticle>::iterator p = physics->particles().begin(); p != physics->particles().end(); ++p)
	{
		// save all parameters which would be set in the constructor of UnifiedParticle
		float px = p->position_[0];
		float py = p->position_[1];
		float pz = p->position_[2];
		float restDens = p->rest_density_;
		float mass = p->particle_mass_;
		float type = p->type_;


		outfileParticles << px << " " << py << " " << pz  << " "
			<< restDens << " "
			<< mass << " "
			<< type << " "
			<< std::endl;
	}
	outfileParticles << std::endl;

	outfileParticles.close();
	++fileCounter;
}

//--------------------------------------------------------------------
Scene::ParticleObject::ParticleObject() : particles(new std::vector<UnifiedParticle>()), solid(false)
	//--------------------------------------------------------------------
{
	positionOffset.set(0.0f, 0.0f, 0.0f);
}

//--------------------------------------------------------------------
Scene::ParticleObject::ParticleObject(std::vector<UnifiedParticle>* partVector) : solid(false)
	//--------------------------------------------------------------------
{
	positionOffset.set(0.0f, 0.0f, 0.0f);
	if(partVector)
	{
		particles = partVector;
	}
	else
	{
		particles = new std::vector<UnifiedParticle>();
	}
}

//--------------------------------------------------------------------
Scene::ParticleObject::ParticleObject(const Scene::ParticleObject& rhs) : solid(rhs.solid), positionOffset(rhs.positionOffset)
	//--------------------------------------------------------------------
{
	particles = new std::vector<UnifiedParticle>();
	*particles = *(rhs.particles);
}

//--------------------------------------------------------------------
Scene::ParticleObject& Scene::ParticleObject::operator=(const Scene::ParticleObject& rhs)
	//--------------------------------------------------------------------
{
	solid = rhs.solid;
	positionOffset = rhs.positionOffset;
	*(this->particles) = *(rhs.particles);
	return *this;
}

//--------------------------------------------------------------------
Scene::ParticleObject::~ParticleObject()
	//--------------------------------------------------------------------
{
	delete particles;
}

//--------------------------------------------------------------------
void Scene::ParticleObject::add(const UnifiedParticle& p)
	//--------------------------------------------------------------------
{
	particles->push_back(p);
}

//--------------------------------------------------------------------
void Scene::ParticleObject::setSolid(bool solid)
	//--------------------------------------------------------------------
{
	this->solid = solid;
}

//--------------------------------------------------------------------
bool Scene::ParticleObject::isSolid()
	//--------------------------------------------------------------------
{
	return solid;
}

//--------------------------------------------------------------------
void Scene::ParticleObject::move(const vmml::Vector3f& distance)
	//--------------------------------------------------------------------
{
	positionOffset += distance;
}

//--------------------------------------------------------------------
vmml::Vector3f Scene::ParticleObject::getPosition()
	//--------------------------------------------------------------------
{
	return positionOffset;
}

//--------------------------------------------------------------------
std::vector<UnifiedParticle>& Scene::ParticleObject::getParticles()
	//--------------------------------------------------------------------
{
	return *particles;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createParticles(UnifiedConstants* constants)
	//--------------------------------------------------------------------
{
	ParticleObject result;

	std::cout << "nofParticles: " << result.getParticles().size() << std::endl;

	double frozenMass = 0.03;//0.0118;

	/*/ frozen particles
	// ground
	for(int ix = -1; ix <= numParticlesX+1; ix++)
	{
	float x = -constants->boxLength/2 + ix * spacing;
	for(int iz = -1; iz <= numParticlesZ+1; iz++)
	{	
	float z = -constants->boxWidth/2 + iz * spacing;
	result.add(UnifiedParticle(x, 0, z, 1000, frozenMass, 3));

	// ceiling
	result.add(UnifiedParticle(x, constants->boxHeight, z, 1000, frozenMass, 3));

	}
	}

	// yz plane
	for(int iy = -1; iy <= numParticlesY+1; iy++)
	{
	float y = 0.0 + iy * spacing;
	for(int iz = -1; iz <= numParticlesZ+1; iz++)
	{	
	float z = -constants->boxWidth/2 + iz * spacing;		
	// yz plane right
	result.add(UnifiedParticle(constants->boxLength/2, y, z, 1000, frozenMass, 3));

	// yz plane left
	result.add(UnifiedParticle(-constants->boxLength/2, y, z, 1000, frozenMass, 3));
	}
	}
	// xy plane
	for(int iy = -1; iy <= numParticlesY+1; iy++)
	{
	float y = 0.0 + iy * spacing;
	for(int ix = -1; ix <= numParticlesX+1; ix++)
	{
	float x = -constants->boxLength/2 + ix * spacing;

	// xy plane front
	result.add(UnifiedParticle(x, y, constants->boxWidth/2, 1000, frozenMass, 3));

	// xy plane back
	result.add(UnifiedParticle(x, y, -constants->boxWidth/2, 1000, frozenMass, 3));
	}
	}
	*/
	//cout << "nofParticlesWithFrozens: " << result.getParticles().size() << endl;




	//*
	float spacing  = constants->particleSpacing; //0.022;
	float jitter = 0.5 * spacing;

	// init particles for box
	float halfSpacing = spacing/2;

	int numParticlesX = (int)(constants->boxLength / spacing);
	int numParticlesY = (int)(constants->boxHeight / spacing);
	int numParticlesZ = (int)(constants->boxWidth  / spacing);


	for(int iy = 0; iy <= numParticlesY; iy++)
	{
		float y = 0.0 + iy * spacing;
		for(int ix = 0; ix <= numParticlesX; ix++)
		{
			float x = -constants->boxLength/2 + ix * spacing;
			for(int iz = 0; iz <= numParticlesZ; iz++)
			{	
				float z = -constants->boxWidth/2 + iz * spacing;		
				if(result.getParticles().size() < 1000) 
				{		
					result.add(UnifiedParticle(x+(drand48()-0.5)*jitter, y+(drand48()-0.5)*jitter, z+(drand48()-0.5)*jitter, 1000, 0.0118, 1));
				}
				else
				{
					result.add(UnifiedParticle(x+(drand48()-0.5)*jitter, y+(drand48()-0.5)*jitter, z+(drand48()-0.5)*jitter, 500, 0.0059, 2));
				}
			}
		}
	}

	std::cout << "nofParticles: " << result.getParticles().size() << std::endl;

	// frozen particles
	// ground
	for(int ix = -1; ix <= numParticlesX+1; ix++)
	{
		float x = -constants->boxLength/2 + ix * spacing;
		for(int iz = -1; iz <= numParticlesZ+1; iz++)
		{	
			float z = -constants->boxWidth/2 + iz * spacing;		
			result.add(UnifiedParticle(x, -spacing, z, 1000, 0.0118, 3));

			// ceiling
			result.add(UnifiedParticle(x, constants->boxHeight + spacing, z, 1000, 0.0118, 3));

		}
	}

	// yz plane
	for(int iy = -1; iy <= numParticlesY+1; iy++)
	{
		float y = 0.0 + iy * spacing;
		for(int iz = -1; iz <= numParticlesZ+1; iz++)
		{	
			float z = -constants->boxWidth/2 + iz * spacing;		
			// yz plane right
			result.add(UnifiedParticle(constants->boxLength/2 + spacing, y, z, 1000, 0.0118, 3));

			// yz plane left
			result.add(UnifiedParticle(-constants->boxLength/2 - spacing, y, z, 1000, 0.0118, 3));
		}
	}
	// xy plane
	for(int iy = -1; iy <= numParticlesY+1; iy++)
	{
		float y = 0.0 + iy * spacing;
		for(int ix = -1; ix <= numParticlesX+1; ix++)
		{
			float x = -constants->boxLength/2 + ix * spacing;

			// xy plane front
			result.add(UnifiedParticle(x, y, constants->boxWidth/2 + spacing, 1000, 0.0118, 3));

			// xy plane back
			result.add(UnifiedParticle(x, y, -constants->boxWidth/2 - spacing, 1000, 0.0118, 3));
		}
	}

	//cout << "nofParticlesWithFrozens: " << result.getParticles().size() << endl;
	//*/
	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::creatElasticSheet(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter)
	//--------------------------------------------------------------------
{
	ParticleObject result;

	const float offsetX = 0.1f;
	const float offsetY = 0.3f;
	const float offsetZ = 0.0f;

	int numParticlesX = (int)(constants->sheetLength / spacing);
	int numParticlesY = 1;
	int numParticlesZ = (int)(constants->sheetWidth  / spacing);

	for(int iy = 0; iy <= numParticlesY; iy++)
	{
		float y = jitter + iy * spacing;
		for(int ix = 0; ix <= numParticlesX; ix++)
		{
			float x = jitter + ix * spacing;	 
			for(int iz = 0; iz <= numParticlesZ; iz++)
			{
				float z = jitter + iz * spacing;	 
				if(result.getParticles().size() < 10000) 
				{
					vmml::Vector3f pos(x+(drand48()-0.5)*jitter+offsetX, y+offsetY, z+(drand48()-0.5)*jitter+offsetZ);
					UnifiedParticle particle(pos[0], pos[1], pos[2], 1000.0f, constants->initialMass, 1);
					// get linearized z-index for integer position within bbox
					particle.index_ = zIndex.CalcIndex(fc->scales * (pos - fc->zindexStartingVec) );

					result.add(particle);
				}
			}
		}
	}

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createChromeSphere(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset)
	//--------------------------------------------------------------------
{
	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	float sphere_radius = 0.075f;

	int particlesPerSide = 300;
	for(int x=-particlesPerSide/2; x<particlesPerSide/2; ++x)
	{
		for(int y=-particlesPerSide/2; y<particlesPerSide/2; ++y)
		{
			for(int z=-particlesPerSide/2; z<particlesPerSide/2; ++z)
			{
				vmml::Vector3f position(x*spacing,y*spacing,z*spacing);
				if(position.length()<sphere_radius)
				{
					vmml::Vector3f pos(position[0]+(drand48()-0.5)*jitter + offsetX, position[1]+sphere_radius + offsetY, position[2]+(drand48()-0.5)*jitter + offsetZ);
					UnifiedParticle particle(pos[0], pos[1], pos[2], 1000.0f, constants->initialMass, 1);

					// get linearized z-index for integer position within bbox
					particle.index_ = zIndex.CalcIndex(fc->scales * (pos - fc->zindexStartingVec) );

					particle.velocity_ = (0.0, 0.0, 0.0);

					result.add(particle);
				}
			}
		}
	}

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createFluidObject(UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset)
	//--------------------------------------------------------------------
{
	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	vmml::Vector3f minBoxVec = fc->fluidBox.getMin();
	vmml::Vector3f maxBoxVec = fc->fluidBox.getMax();
	vmml::Vector3f delta = maxBoxVec - minBoxVec;
	vmml::Vector3i numParticles = delta / vmml::Vector3f(spacing);

	int nofParticles = numParticles.x * numParticles.y * numParticles.z;

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);
	int counter = 0;
	for(int iy = 0; iy < numParticles.y; iy++) {
		float y = jitter + iy * spacing;

		for(int ix = 0; ix < numParticles.x; ix++) {
			float x = jitter + ix * spacing;

			for(int iz = 0; iz < numParticles.z; iz++) {	
				float z = jitter + iz * spacing;		

				if(counter > nofParticles)
					continue;
				// jittered position within bbox
				vmml::Vector3f pos(x+(drand48()-0.5)*jitter+offsetX, y+(drand48()-0.5)*jitter+offsetY, z+(drand48()-0.5)*jitter+offsetZ);
				UnifiedParticle particle(pos[0], pos[1], pos[2], 1000.0f, constants->initialMass, 1);

				// get linearized z-index for integer position within bbox
				particle.index_ = zIndex.CalcIndex(fc->scales * (pos - fc->zindexStartingVec) );

				particle.velocity_ = (0.0, 0.0, 0.0);

				result.add(particle);

				counter++;
			}
		}
	}
	//assert( counter == particles.size() );

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::readFluidObjectFromFile(const char* fileName, UnifiedPhysics* physics, UnifiedConstants* constants, int& numParticles)
	//--------------------------------------------------------------------
{
	ParticleObject result;
	int nofParticles = physics->my_unified_io()->ReadInFluidParticleNum(fileName);
	numParticles = nofParticles;

	float* pos				 =  new float[nofParticles*3];
	float* vel				 =  new float[nofParticles*3];
	int*   type				 =  new int[nofParticles];
	float* corr_pressure	 =  new float[nofParticles];
	float* predicted_density =  new float[nofParticles];
	physics->my_unified_io()->ReadInFluidParticlesPCISPH(fileName, pos, vel, type, corr_pressure, predicted_density);

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);

	for(int i = 0; i < nofParticles; ++i) {	
		float posX = pos[i*3];
		float posY = pos[i*3+1];
		float posZ = pos[i*3+2];
		vmml::Vector3f p_pos(posX, posY, posZ);

		int   p_type = type[i];

		float velX = vel[i*3];
		float velY = vel[i*3+1];
		float velZ = vel[i*3+2];
		vmml::Vector3f p_vel(velX, velY, velZ);

		float p_corr_pressure = corr_pressure[i];

		float p_predicted_density = predicted_density[i];		

		UnifiedParticle particle(p_pos, p_type, p_vel, p_corr_pressure, p_predicted_density, 1000.0f, constants->initialMass);

		// get linearized z-index for integer position within bbox
		particle.index_ = zIndex.CalcIndex(fc->scales * (p_pos - fc->zindexStartingVec) );

		result.add(particle);
	}

	delete[] predicted_density;
	delete[] corr_pressure;
	delete[] type;
	delete[] vel;
	delete[] pos;

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createRigidObject(const BBox& rigidBox, UnifiedPhysics* physics, UnifiedConstants* constants, float spacing, float jitter, const vmml::Vector3f &offset)
	//--------------------------------------------------------------------
{
	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	vmml::Vector3f delta = rigidBox.getMax()-rigidBox.getMin();
	vmml::Vector3i numParticles = delta / vmml::Vector3f(spacing);

	const int nofParticles = numParticles.x * numParticles.y * numParticles.z;

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);

	int counter = 0;
	for(int iy = 0; iy < numParticles.y; iy++) {
		float y = jitter + iy * spacing;

		for(int ix = 0; ix < numParticles.x; ix++) {
			float x = jitter + ix * spacing;

			for(int iz = 0; iz < numParticles.z; iz++) {	
				float z = jitter + iz * spacing;		

				if(counter > nofParticles)
					continue;

				// jittered position within bbox
				vmml::Vector3f pos(x+(drand48()-0.5)*jitter + offsetX, y+(drand48()-0.5)*jitter + offsetY, z+(drand48()-0.5)*jitter + offsetZ);
				UnifiedParticle particle(pos[0], pos[1], pos[2], 1000.0f, constants->initialMass, 1);

				// get linearized z-index for integer position within bbox
				particle.index_ = zIndex.CalcIndex(fc->scales * (pos - fc->zindexStartingVec) );

				particle.velocity_ = (0.0, 0.0, 0.0);

				result.add(particle);

				counter++;
			}
		}
	}
	//assert( counter == particles.size() );

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createSurfaceParticlesObject(const std::string filename, const vmml::Vector3f scale, UnifiedPhysics* physics, UnifiedConstants* constants, 
	float radius, float jitter, const vmml::Vector3f &offset, float density, float rb_density)
	//--------------------------------------------------------------------
{
	const float density_ratio = rb_density/constants->fluidRestDensity;

	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	// extract particles from triangle mesh
	TriangleMesh tmesh(filename);
	tmesh.Scale( Vector3DMeshVersion(scale.x, scale.y, scale.z) ); //expand the mesh
	std::vector< Particle > temp_particles;
	ParticlePlacer::CoverMeshSurfaceWithParticles(tmesh, temp_particles, radius, density);

	const int nofParticles = temp_particles.size();

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);

	int counter = 0;
	for(int i = 0; i < nofParticles; ++i) {	
		if(counter > nofParticles)
			continue;

		// jittered position within bbox
		Vector3DMeshVersion location = temp_particles[i].Location();
		vmml::Vector3f pos(location[0] + offsetX, location[1] + offsetY, location[2] + offsetZ);
		UnifiedParticle particle(pos[0], pos[1], pos[2], rb_density, constants->initialMass * density_ratio, 1);

		// get linearized z-index for integer position within bbox
		particle.index_ = zIndex.CalcIndex(fc->scales * (pos - fc->zindexStartingVec) );

		particle.velocity_ = (0.0, 0.0, 0.0);

		result.add(particle);

		counter++;
	}

	return result;
}

void Scene::CreateFullParticleObject(UnifiedConstants* fc, const vmml::Vector3f &offset, const float density, const int particleType, const char* model_file_name, 
	const vmml::Vector3f &scale, const float particle_spacing, bool isRigidBody)
{

	UnifiedParticle templateParticle(0.0, 0.0, 0.0, density, fc->initialMass, particleType);
	ParticleObject rigid_box_objectBunny = LoadModel(&ObjLoader::GetInstance(), model_file_name, scale.x, scale.y, scale.z, particle_spacing, templateParticle);
	const uint numParticles1 = rigid_box_objectBunny.getParticles().size();
	fc->numRigidParticles += numParticles1;
	assert(numParticles1 <= MAX_NUM_RIGID_PARTICLES);
	if (particleType == 0 || particleType == 6)
	{
		rigid_box_objectBunny.setSolid(true);
	}		
	for(std::vector<UnifiedParticle>::iterator p=rigid_box_objectBunny.getParticles().begin(); p != rigid_box_objectBunny.getParticles().end(); ++p)
	{
		p->position_ += vmml::Vector3f(offset.x, offset.y, offset.z);
		p->density_ = density;
		p->particle_mass_ = fc->initialMass * density/fc->fluidRestDensity;
		p->index_ = zIndex.CalcIndex(fc->scales * (p->position_ - fc->zindexStartingVec) );
		p->gas_const_ = 50;
		p->rigidbody_ = isRigidBody;
		p->type_ = particleType;
	}
	add(rigid_box_objectBunny);
	//*/
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createParticlesPeriodicBoundary(UnifiedConstants* constants)
	//--------------------------------------------------------------------
{
	ParticleObject result;

	float spacing  = 0.020; //0.022;
	float jitter = 0.5 * spacing;

	// init particles for box
	float halfSpacing = spacing/2;

	int numParticlesX = (int)(constants->boxLength / spacing);
	int numParticlesY = (int)(constants->boxHeight / spacing);
	int numParticlesZ = (int)(constants->boxWidth  / spacing);


	if(numParticlesX <= 0)
		numParticlesX = 1;
	if(numParticlesY <= 0)
		numParticlesY = 1;
	if(numParticlesZ <= 0)
		numParticlesZ = 1;

	for(int iy = 0; iy < numParticlesY; iy++)
	{
		float y = 0.0 + iy * spacing;
		for(int ix = 0; ix < numParticlesX; ix++)
		{
			float x = -constants->boxLength/2 + halfSpacing + ix * spacing;
			for(int iz = 0; iz < numParticlesZ; iz++)
			{	
				float z = -constants->boxWidth/2 + halfSpacing + iz * spacing;		
				if(result.getParticles().size() < 9300) //9300
				{										
					result.add(UnifiedParticle(x+(drand48()-0.5)*jitter, y+(drand48()-0.5)*jitter, z+(drand48()-0.5)*jitter, 800, 0.00944, 1));
				}
				else
				{
					result.add(UnifiedParticle(x+(drand48()-0.5)*jitter, y+(drand48()-0.5)*jitter, z+(drand48()-0.5)*jitter, 700, 0.00826, 2));
				}
			}
		}
	}

	std::cout << "nofParticles: " << result.getParticles().size() << std::endl;

	// frozen particles
	// ground
	for(int ix = 0; ix < numParticlesX; ix++)
	{
		float x = -constants->boxLength/2 + halfSpacing + ix * spacing;
		for(int iz = 0; iz < numParticlesZ; iz++)
		{	
			float z = -constants->boxWidth/2 + halfSpacing + iz * spacing;		

			result.add(UnifiedParticle(x, -spacing, z, 1000, 0.0118, 3));

		}
	}
	//cout << "nofParticlesWithFrozens: " << result.getParticles().size() << endl;

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createBoxBoundaryFrozenParticles(UnifiedPhysics* physics, float spacing, UnifiedConstants* fc, float jitter, const vmml::Vector3f &offset)
	//--------------------------------------------------------------------
{
	//NOTE: all particles are displayed according to the realBoxContainer

	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	// distance from boundary to frozen particles (also the distance from virtual boundary to static frozen ones)
	float distance = fc->globalSupportRadius - fc->distance_frozen_to_boundary; 

	vmml::Vector3f minBox = fc->realBoxContainer.getMin() - vmml::Vector3f(fc->distance_frozen_to_boundary);
	vmml::Vector3f maxBox = fc->realBoxContainer.getMax() + vmml::Vector3f(fc->distance_frozen_to_boundary);
	vmml::Vector3f delta = maxBox - minBox;

	vmml::Vector3i numParticles = delta / vmml::Vector3f(spacing);

	int nofParticles = numParticles.x * numParticles.y * numParticles.z;

	int numParticlesX = numParticles.x;
	int numParticlesY = numParticles.y;
	int numParticlesZ = numParticles.z;

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);

	UnifiedParticle particle_template(0.0f, 0.0f, 0.0f, 1000, 0.03, 3);

	vmml::Vector3f relativePos;
	//* ground & ceiling (xz planes)
	for(int x = 0; x < numParticlesX + 2; ++x) 
	{
		for(int z = 0; z < numParticlesZ + 2; ++z)
		{
			// ground
			particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance,fc->zindexStartingVec.z + distance + z*spacing);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}		
			particle_template.velocity_ = (0.0, 0.0, 0.0);
			particle_template.normalized_normal_ = vmml::Vector3f(0.0, 1.0, 0.0);
			result.add(particle_template);

			// ceiling
			particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + delta.y, fc->zindexStartingVec.z + distance + z*spacing);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}
			particle_template.normalized_normal_ = vmml::Vector3f(0.0, -1.0, 0.0);
			result.add(particle_template);
		}
	}
	//*/

	//* front & back wall (xy planes)
	for(int x = 0; x < numParticlesX + 2; ++x)
	{
		for(int y = 1; y < numParticlesY + 1; ++y)
		{
			// back wall
			particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}
			particle_template.velocity_ = (0.0, 0.0, 0.0);
			particle_template.normalized_normal_ = vmml::Vector3f(0.0, 0.0, 1.0);
			result.add(particle_template);

			// front wall
			particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + delta.z);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}
			particle_template.velocity_ = (0.0, 0.0, 0.0);
			particle_template.normalized_normal_ = vmml::Vector3f(0.0, 0.0, -1.0);
			result.add(particle_template);
		}
	}
	//*/

	//* left & right wall (yz planes)
	for(int y = 1;y < numParticlesY + 1; ++y)
	{
		for(int z = 1;z < numParticlesZ + 1; ++z)
		{
			// left wall
			particle_template.position_.set(fc->zindexStartingVec.x + distance, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + z*spacing);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}
			particle_template.velocity_ = (0.0, 0.0, 0.0);
			particle_template.normalized_normal_ = vmml::Vector3f(1.0, 0.0, 0.0);
			result.add(particle_template);

			// right wall
			particle_template.position_.set(fc->zindexStartingVec.x + distance + delta.x, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + z*spacing);
			relativePos = particle_template.position_ - fc->zindexStartingVec;
			if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
			{
				particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
			}
			else
			{
				std::cerr << "relativePos should be larger than 0!!!" << std::endl;
				abort();
			}
			particle_template.velocity_ = (0.0, 0.0, 0.0);
			particle_template.normalized_normal_ = vmml::Vector3f(-1.0, 0.0, 0.0);
			result.add(particle_template);
		}
	}
	//*/

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createBoxBoundaryFrozenParticlesMultiLayers(UnifiedPhysics* physics, float spacing, UnifiedConstants* fc, float jitter, const vmml::Vector3f &offset)
	//--------------------------------------------------------------------
{
	//NOTE: all particles are displayed according to the realBoxContainer

	const float offsetX = offset[0];
	const float offsetY = offset[1];
	const float offsetZ = offset[2];

	ParticleObject result;

	const int numLayers = 2;

	// distance from boundary to frozen particles (also the distance from virtual boundary to static frozen ones)
	float distance = fc->globalSupportRadius - fc->distance_frozen_to_boundary; 

	vmml::Vector3f minBox = fc->realBoxContainer.getMin() - vmml::Vector3f(fc->distance_frozen_to_boundary);
	vmml::Vector3f maxBox = fc->realBoxContainer.getMax() + vmml::Vector3f(fc->distance_frozen_to_boundary);
	vmml::Vector3f delta = maxBox - minBox;

	vmml::Vector3i numParticles = delta / vmml::Vector3f(spacing);

	int nofParticles = numParticles.x * numParticles.y * numParticles.z;

	int numParticlesX = numParticles.x;
	int numParticlesY = numParticles.y;
	int numParticlesZ = numParticles.z;

	std::vector<UnifiedParticle>& tempParticles = result.getParticles();
	tempParticles.reserve(nofParticles);

	UnifiedParticle particle_template(0.0f, 0.0f, 0.0f, 1000, 0.03, 3);

	vmml::Vector3f relativePos;
	//* ground & ceiling (xz planes)
	for(int x = 0; x < numParticlesX + 2; ++x) 
	{
		for(int z = 0; z < numParticlesZ + 2; ++z)
		{
			for (int y = 0; y < numLayers; ++y)
			{
				// ground
				particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance - y*spacing,fc->zindexStartingVec.z + distance + z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}		
				particle_template.velocity_ = (0.0, 0.0, 0.0);
				particle_template.normalized_normal_ = vmml::Vector3f(0.0, 1.0, 0.0);
				result.add(particle_template);

				// ceiling
				particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + delta.y + y*spacing, fc->zindexStartingVec.z + distance + z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}
				particle_template.normalized_normal_ = vmml::Vector3f(0.0, -1.0, 0.0);
				result.add(particle_template);
			}

		}
	}
	//*/

	//* front & back wall (xy planes)
	for(int x = 0; x < numParticlesX + 2; ++x)
	{
		for(int y = 1; y < numParticlesY + 1; ++y)
		{
			for (int z = 0; z < numLayers; ++z)
			{
				// back wall
				particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance - z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}
				particle_template.velocity_ = (0.0, 0.0, 0.0);
				particle_template.normalized_normal_ = vmml::Vector3f(0.0, 0.0, 1.0);
				result.add(particle_template);

				// front wall
				particle_template.position_.set(fc->zindexStartingVec.x + distance + x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + delta.z + z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}
				particle_template.velocity_ = (0.0, 0.0, 0.0);
				particle_template.normalized_normal_ = vmml::Vector3f(0.0, 0.0, -1.0);
				result.add(particle_template);
			}

		}
	}
	//*/

	//* left & right wall (yz planes)
	for(int y = 1;y < numParticlesY + 1; ++y)
	{
		for(int z = 1;z < numParticlesZ + 1; ++z)
		{
			for (int x = 0; x < numLayers; ++x)
			{
				// left wall
				particle_template.position_.set(fc->zindexStartingVec.x + distance - x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}
				particle_template.velocity_ = (0.0, 0.0, 0.0);
				particle_template.normalized_normal_ = vmml::Vector3f(1.0, 0.0, 0.0);
				result.add(particle_template);

				// right wall
				particle_template.position_.set(fc->zindexStartingVec.x + distance + delta.x + x*spacing, fc->zindexStartingVec.y + distance + y*spacing, fc->zindexStartingVec.z + distance + z*spacing);
				relativePos = particle_template.position_ - fc->zindexStartingVec;
				if (relativePos.x > 0.0 && relativePos.y > 0.0 && relativePos.z > 0.0)
				{
					particle_template.index_ = zIndex.CalcIndex(fc->scales * relativePos);
				}
				else
				{
					std::cerr << "relativePos should be larger than 0!!!" << std::endl;
					abort();
				}
				particle_template.velocity_ = (0.0, 0.0, 0.0);
				particle_template.normalized_normal_ = vmml::Vector3f(-1.0, 0.0, 0.0);
				result.add(particle_template);
			}

		}
	}
	//*/

	return result;
}

//--------------------------------------------------------------------
Scene::ParticleObject Scene::createFrozenParticlesAtBottom(float spacing, UnifiedConstants* constants)
	//--------------------------------------------------------------------
{
	ParticleObject result;
	int numParticlesX = (int)(constants->boxLength / spacing)+1;
	int numParticlesY = (int)(constants->boxHeight / spacing)+1;
	int numParticlesZ = (int)(constants->boxWidth  / spacing)+1;
	float distance = constants->globalSupportRadius*0.3; // distance from boundary to frozens

	// ground
	for(int x=0;x<numParticlesX;++x)
	{
		for(int z=0;z<numParticlesZ;++z)
		{
			result.add(UnifiedParticle(-constants->boxLength/2 + x*spacing, -distance, -constants->boxWidth/2 + z*spacing, 1000, 0.03, 3));
		}
	}
	return result;
}
