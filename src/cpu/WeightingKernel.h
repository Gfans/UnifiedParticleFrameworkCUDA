#ifndef GPU_UNIFIED_WEIGHTING_KERNEL_H_
#define GPU_UNIFIED_WEIGHTING_KERNEL_H_

#include "UnifiedConstants.h"

class WeightingKernel
{
public:

	WeightingKernel(UnifiedConstants *fc);
	~WeightingKernel();	

	static int lut_size() { return kLutSize; }
	float* lut_kernel_m4() { return &lut_kernel_m4_[0]; }
	float* lut_kernel_pressure_grad() { return &lut_kernel_pressure_grad_[0]; }
	float* lut_kernel_viscosity_lap() { return &lut_kernel_viscosity_lap_[0]; }
	float* lut_spline_surface_tension() { return &lut_spline_surface_tension_[0]; }
	float* lut_spline_surface_adhesion() { return &lut_spline_surface_adhesion_[0]; }

	float KernelDensity(float distSq, float sr);  
	float KernelDensityGrad(float dist, float sr);	
	float KernelNew(float distSq, float sr);
	float KernelNewGrad(float dist, float sr);		

	// M4 a.k.a cubic spline
	float KernelM4(float dist, float sr);
	float KernelM4Norm(float s);
	float kernelM4Lut(float dist); 

	float kernelM4Grad(float dist, float sr);
	float kernelPressure(float dist, float sr);		

	float kernelPressureGrad(float dist, float sr);	
	float kernelPressureGradLut(float dist);

	float kernelViscosityLaplacian(float dist, float sr);
	float kernelViscosityLaplacianLut(float dist);

	float kernelSplineSurfaceTension(float dist, float sr);
	float kernelSplineSurfaceTensionLut(float dist);

	float kernelSplineSurfaceAdhesion(float dist, float sr);
	float kernelSplineSurfaceAdhesionLut(float dist);

	void  kernelSurfaceTensionOldMethod(float distSq, float& grad, float &lap, float &clapSelf, float sr);

private:

	void initLookupTables();

	void  calculateKernelConstants();	

private:

	float kernel_density_const_;
	float kernel_pressure_const_;
	float kernel_viscosity_const_;
	float kernel_surface_tension_const_;

	float support_radius_sq_;

	static int const kLutSize = 300; 
	float lut_kernel_m4_[kLutSize+1];
	float lut_kernel_pressure_grad_[kLutSize+1];
	float lut_kernel_viscosity_lap_[kLutSize+1];

	UnifiedConstants *fc_;

	float lut_spline_surface_tension_[kLutSize+1];
	float lut_spline_surface_adhesion_[kLutSize+1];
};

#endif	// GPU_UNIFIED_WEIGHTING_KERNEL_H_
