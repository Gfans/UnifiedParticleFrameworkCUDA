#include <iomanip>
#include "UnifiedIO.h"

UnifiedIO::UnifiedIO()
{

}

UnifiedIO::~UnifiedIO()
{

}

void UnifiedIO::SaveParticlePositions(const std::vector<UnifiedParticle> &particles)
{
	static int fileCounter = 0;
	char filename[200];
	sprintf(filename, "K:/SimulationData/output/particles%04d.txt", fileCounter);
	std::cout << filename << std::endl;
	out_file_particles_.open(filename);
	out_file_particles_.clear();

	// write out the total number of particles
	out_file_particles_ << particles.size() << std::endl;
	std::cout << particles.size() << std::endl;

	for (std::vector<UnifiedParticle>::const_iterator p = particles.begin(); p != particles.end(); ++p)
	{
		// save all parameters which would be set in the constructor of UnifiedParticle
		float px = p->position_[0];
		float py = p->position_[1];
		float pz = p->position_[2];

		float vx = 0;
		float vy = 0;
		float vz = 0;
		float sr = 0;
		float gasConst = 0;	
		float restDens = 0;
		float dens = 0;
		float mass = 0;

		out_file_particles_.precision(20);

		out_file_particles_ << px << " " << py << " " << pz  << " "
			<< restDens << " "
			<< mass << " "
			<< dens << " "
			<< vx << " " << vy << " " << vz << " " 
			<< sr << " " 
			<< gasConst << " " 
			<< std::endl;
	}

	out_file_particles_.close();
	++fileCounter;
}

void UnifiedIO::SaveLiquidParticleInfoFull(float* pos, int* type, int num_rounded_particles, int num_liquid_particles)
{
	static int fileCounter = 0;
	char filename[200];
	sprintf(filename, "K:/SimulationData/output/FluidParticles/particles%04d.txt", fileCounter);
	std::cout << filename << std::endl;
	out_file_particles_.open(filename);
	out_file_particles_.clear();

	// write out the total number of particles
	out_file_particles_ << num_liquid_particles << std::endl;

	for (int i = 0; i < num_rounded_particles; ++i)
	{
		int ptype = type[i];

		if (ptype == 2)	// we only save liquid particle's pos info for fluid rendering
		{
			float px = pos[4*i];
			float py = pos[4*i+1];
			float pz = pos[4*i+2];

			int type = ptype;
			float vx = 0;
			float vy = 0;
			float vz = 0;
			float gasConst = 0;	
			float restDens = 0;
			float dens = 0;
			float mass = 0;

			out_file_particles_.precision(8);

			out_file_particles_ << setiosflags(std::ios::left) << std::setw(15) << px << " " 
				<< setiosflags(std::ios::left) << std::setw(15) << py << " " 
				<< setiosflags(std::ios::left) << std::setw(15) << pz  << " "							 
				<< setiosflags(std::ios::left) << std::setw(3) << ptype << " "
				<< setiosflags(std::ios::left) << std::setw(15) << vx << " " << vy << " " << vz << " "  
				<< setiosflags(std::ios::left) << std::setw(2) << restDens << " "
				<< setiosflags(std::ios::left) << std::setw(2) << mass << " "
				<< setiosflags(std::ios::left) << std::setw(2) << dens << " "
				<< setiosflags(std::ios::left) << std::setw(2) << gasConst << " " 
				<< std::endl;
		}
	}

	out_file_particles_.close();
	++fileCounter;	
}

void UnifiedIO::SaveRigidParticleInfoFull(float* pos, int* type, int num_rounded_particles, int num_liquid_particles)
{
	static int fileCounter = 0;
	char filename[200];
	sprintf(filename, "K:/SimulationData/output/RigidParticles/rigid1Particles%04d.txt", fileCounter);
	std::cout << filename << std::endl;
	out_file_particles_.open(filename);
	out_file_particles_.clear();

	// write out the total number of particles
	out_file_particles_ << num_liquid_particles << std::endl;

	for (int i = 0; i < num_rounded_particles; ++i)
	{
		int ptype = type[i];

		if (ptype == 0 || ptype == 6)	// we only save liquid particle's pos info for fluid rendering
		{
			float px = pos[4*i];
			float py = pos[4*i+1];
			float pz = pos[4*i+2];

			int type = ptype;
			float vx = 0;
			float vy = 0;
			float vz = 0;
			float gasConst = 0;	
			float restDens = 0;
			float dens = 0;
			float mass = 0;

			out_file_particles_.precision(8);

			out_file_particles_ << std::setiosflags(std::ios::left) << std::setw(15) << px << " " 
				<< std::setiosflags(std::ios::left) << std::setw(15) << py << " " 
				<< std::setiosflags(std::ios::left) << std::setw(15) << pz  << " "							 
				<< std::setiosflags(std::ios::left) << std::setw(3) << ptype << " "
				<< std::setiosflags(std::ios::left) << std::setw(15) << vx << " " << vy << " " << vz << " "  
				<< std::setiosflags(std::ios::left) << std::setw(2) << restDens << " "
				<< std::setiosflags(std::ios::left) << std::setw(2) << mass << " "
				<< std::setiosflags(std::ios::left) << std::setw(2) << dens << " "
				<< std::setiosflags(std::ios::left) << std::setw(2) << gasConst << " " 
				<< std::endl;
		}
	}

	out_file_particles_.close();
	++fileCounter;	
}

int UnifiedIO::ReadInFluidParticleNum(const char* fileName)
{
	try 
	{
		in_file_particles_.open(fileName);
	}
	catch(...)
	{
		std::cout << "Error opening file" << std::endl;
	}

	int cnt;
	if(in_file_particles_)
	{
		in_file_particles_ >> cnt;
	}
	else
	{
		std::cerr << "wrong filename!" << std::endl; 
		exit(-1);
	}

	in_file_particles_.close();

	return cnt;	
}

void UnifiedIO::ReadInFluidParticlesPCISPH(const char* fileName, float* pos, float* vel, int* type, float* corr_pressure, float* predicted_density)
{
	try 
	{
		in_file_particles_.open(fileName);
	}
	catch(...)
	{
		std::cout << "Error opening file" << std::endl;
	}	

	int cnt;
	if (in_file_particles_)
	{
		in_file_particles_ >> cnt;
	} 
	else
	{
		std::cerr << "wrong particle number!" << std::endl; 
		exit(-1);
	}

	int readCnt = 0;
	while(in_file_particles_ && readCnt < cnt)
	{
		in_file_particles_ >> pos[3*readCnt] >> pos[3*readCnt+1] >> pos[3*readCnt+2]
		>> type[readCnt]
		>> vel[3*readCnt] >> vel[3*readCnt+1] >> vel[3*readCnt+2]
		>> corr_pressure[readCnt]
		>> predicted_density[readCnt];
		++readCnt;
	}

	in_file_particles_.close();
}
