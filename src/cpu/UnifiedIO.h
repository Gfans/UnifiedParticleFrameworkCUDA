#ifndef GPU_UNIFIED_IO_H_
#define GPU_UNIFIED_IO_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>

#include "UnifiedParticle.h"
#include "vmmlib/vector3.h"

class UnifiedIO
{
public:
	UnifiedIO();
	~UnifiedIO();

	void SaveParticlePositions(const vector<UnifiedParticle>& particles);	
	void SaveLiquidParticleInfoFull(float* pos, int* type, int num_rounded_particles, int num_liquid_particles);
	void SaveRigidParticleInfoFull(float* pos, int* type, int num_rounded_particles, int num_liquid_particles);

	int  ReadInFluidParticleNum(const char* fileName);
	void ReadInFluidParticlesPCISPH(const char* fileName, float* pos, float* vel, int* type, float* corr_pressure, float* predicted_density);

private:	
	std::ofstream out_file_particles_;
	std::ifstream in_file_particles_;

	static const int kMaxNum = 100;
	char m_buffer[kMaxNum];

};

#endif	// GPU_UNIFIED_IO_H_




