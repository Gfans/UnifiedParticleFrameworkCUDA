#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <new>
#include <cmath>
#include <cuda_runtime.h>
#include <windows.h>

#include "UnifiedConstants.h"
#include "UnifiedParticle.h"
#include "UnifiedPhysics.h"
#include "particles.cuh"
#include "radixsort.cuh"
#include "WeightingKernel.h"
#include "zIndex.h"
#include "GlutWindow.h"
#include "UnifiedWindow.h"
#include "GlutApplication.h"
#include "global.h"

UnifiedWindow myWindow;

int main(int argc, char** argv)
{
	// create GLUT graphics window
	GlutApplication myApp("Unified Particle Framework CUDA", &myWindow);

	std::cout << "----------------------CUDA Device Information!----------------------" << std::endl;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "CUDA-Capable device count: " << deviceCount << std::endl;
	printf( "\n" );
	// Device Enumeration
	for (int device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf( "------ General Information for device %d ------\n", device );
		printf( "Name:  %s\n", deviceProp.name );
		printf( "Compute capability:  %d.%d\n", deviceProp.major, deviceProp.minor );	
		printf( "\n" );
	}

	// Device Selection
	bool useGTX780 = true;
	if (useGTX780)
	{
		cudaSetDevice(0);
		std::cout << "\nCurrent device is set to GTX 780!" << std::endl << std::endl;
	}
	else
	{
		cudaSetDevice(1); // use GTX 480
		std::cout << "\nCurrent device is set to GTX 480!" << std::endl << std::endl;
	}

	fc = new UnifiedConstants();
	myFluid = new UnifiedPhysics(fc); 

	myApp.run(); // calls glutMainLoop()

	delete myFluid;
	myFluid = NULL;

	delete fc;
	fc = NULL;

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	return 0;
}