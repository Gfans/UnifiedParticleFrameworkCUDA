#include <math.h>

#include "WeightingKernel.h"

const double kPi = 3.1415926535897932384626;

WeightingKernel::WeightingKernel(UnifiedConstants *_fc)
	: fc_(_fc)
{
	initLookupTables();
}

WeightingKernel::~WeightingKernel()
{

}

float WeightingKernel::KernelDensity(float distSq, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	// Normalized 6th order spline: W(r,h) = 315/(64*pi*h^9)*(h^2 - r^2)^3;
	//float f = support_radius_sq_ - distSq;

	float srSQ = sr * sr;
	if(distSq > srSQ)
		return 0.0f;
	float srSqSq = srSQ * srSQ;
	kernel_density_const_ = 1.56668147f * 1.0f/(srSqSq*srSqSq*sr);	
	float f = srSQ - distSq;
	return kernel_density_const_ * f * f * f;
}

float WeightingKernel::KernelDensityGrad(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < supportRadius !!

	// grad W(r,h) = -945 / (32pi*h^9) * (h≤-r≤)^2 * r
	// returns only the factor that r has to be scaled to become the gradient
	if(dist > sr)
		return 0.0f;

	support_radius_sq_ = sr*sr;
	kernel_density_const_ = 315.0f/(64.0f * (float)kPi * sr*sr*sr*sr*sr*sr*sr*sr*sr);
	float distSq = dist * dist;
	float f = support_radius_sq_ - distSq;
	return -6.0 * kernel_density_const_ * f * f;
}

float WeightingKernel::KernelNew(float distSq, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	// Normalized 6th order spline: W(r,h) = 315/(64*pi*h^9)*(h^2 - r^2)^3;

	//float f = support_radius_sq_ - distSq;
	if(distSq > sr*sr)
		return 0.0f;
	kernel_density_const_ = 3465.0f/(512.0f * (float)kPi * sr*sr*sr);	
	float f = 1.0f - distSq/(sr*sr);
	return kernel_density_const_ * f * f * f * f;
}

float WeightingKernel::KernelNewGrad(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	// Normalized 6th order spline: W(r,h) = 315/(64*pi*h^9)*(h^2 - r^2)^3;

	//float f = support_radius_sq_ - distSq;
	if(dist > sr)
		return 0.0f;
	kernel_density_const_ = 3465.0f/(512.0f * (float)kPi * sr*sr*sr*sr*sr);	
	float f = 1.0f - (dist*dist)/(sr*sr);
	return -8.0f * kernel_density_const_ * f * f * f;
}

float WeightingKernel::KernelM4(float dist, float sr)
{
	float s = dist / sr;
	float result;
	float factor = 2.546479089470325472f / (sr * sr * sr);
	if(dist < 0.0f || dist >= sr)
	{
		return 0.0f;
	}
	else
	{
		if(s < 0.5f)
		{
			result = 1.0f - 6.0 * s * s + 6.0f * s * s * s;
		}
		else
		{
			float tmp = 1.0f - s;
			result = 2.0 * tmp * tmp * tmp;
		}
	}
	return factor * result;
}

float WeightingKernel::KernelM4Norm(float s)
{
	// this gives kernel values between 0 and 1

	float result;

	if(s < 0.0f || s > 1.0)
		return 0.0f;
	else
	{
		if(s < 0.5f)
		{
			result = 1.0f - 6.0 * s * s + 6.0f * s * s * s;
		}
		else
		{
			float tmp = 1.0f - s;
			result = 2.0 * tmp * tmp * tmp;
		}
	}
	return result;
}

float WeightingKernel::kernelM4Lut(float dist)
{
	static float factor = 1.0 / fc_->globalSupportRadius;
	int index = dist * factor * kLutSize;

	return (index >= kLutSize) ? 0.0 : lut_kernel_m4_[index];
}

float WeightingKernel::kernelM4Grad(float dist, float sr)
{
	float s = dist / sr;
	float result;
	float factor = 2.546479089470325472f / (sr * sr * sr * sr * sr);
	if(dist < 0.0f || dist >= sr)
	{
		//		cerr << "error! called kernelM4Grad() with a negative distance" << endl;
		//		assert(dist >= 0.0f);
		//		abort();
		return 0.0f;
	}
	else
		if(s < 0.5f)
		{
			result = - 12.0f + 18.0f * s;
		}
		else
		{
			float tmp = 1.0f - s;
			result = -6.0 * tmp * tmp / s;
		}
		return factor * result;
}

float WeightingKernel::kernelPressure(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < supportRadius !!
	// W(r,h) = 15/(pi*h^6)*(h-|r|)^3
	if(dist > sr)
		return 0.0f;
	kernel_pressure_const_ = 15.f/((float(kPi)*sr*sr*sr*sr*sr*sr));
	return kernel_pressure_const_ * (sr-dist)*(sr-dist)*(sr-dist);
}

float WeightingKernel::kernelPressureGrad(float dist, float sr)
{
	if(dist == 0)
		return 0.0f;
	if(dist > sr)
		return 0.0f;

	// NOTE: This method assumes that |rvec| < supportRadius !!
	// W(r,h) = 15/(pi*h^6)*(h-|r|)^3
	// grad W(r,h) = -45 / (pi*h^6*|r|) * (h-|r|)^2 * r
	// returns only the factor that r has to be scaled to become the gradient

	//return kernelPressureConst / dist * (supportRadius-dist)*(supportRadius-dist);
	float kernelPressureConst = -45.f/((float(kPi)*sr*sr*sr*sr*sr*sr));
	return kernelPressureConst / dist * (sr-dist)*(sr-dist);
}

float WeightingKernel::kernelPressureGradLut(float dist)
{
	static float factor = 1.0 / fc_->globalSupportRadius;
	int index = dist * factor * kLutSize;

	return (index >= kLutSize) ? 0.0 : lut_kernel_pressure_grad_[index];
}

float WeightingKernel::kernelViscosityLaplacian(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	// lap W(r,h) = -45 / (pi*h^6) * (h - |r|)	

	//return kernel_viscosity_const_ * (supportRadius - dist);
	if(dist > sr)
		return 0.0f;
	kernel_viscosity_const_ = 45.f/((float(kPi)*sr*sr*sr*sr*sr*sr));
	return kernel_viscosity_const_ * (sr - dist);
}

float WeightingKernel::kernelViscosityLaplacianLut(float dist)
{
	static float factor = 1.0 / fc_->globalSupportRadius;
	int index = dist * factor * kLutSize;

	return (index >= kLutSize) ? 0.0 : lut_kernel_viscosity_lap_[index];
}

float WeightingKernel::kernelSplineSurfaceTension(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	/****************************************************************
				32 / (pi*h^9) * (h - |r|)						,  h/2 < r <= h
	C(r) =		32 / (pi*h^9) *  ( 2 * (h-r)^3 * r^3 - h^6/64)  , 0 < r <= h/2
				0, otherwise
	****************************************************************/

	const float tempConst = 32.0f/((float(kPi)*sr*sr*sr*sr*sr*sr*sr*sr*sr));

	if (dist > 0.5 * sr && dist <= sr)
	{
		return tempConst * (sr - dist) * (sr - dist) * (sr - dist) * dist * dist * dist;
	} 
	else if (dist > 0 && dist <= 0.5 * sr)
	{
		return tempConst * ( 2.0f * (sr - dist) * (sr - dist) * (sr - dist) * dist * dist * dist - sr*sr*sr*sr*sr*sr / 64.0f );
	} 
	else
	{
		return 0.0f;
	}
}

float WeightingKernel::kernelSplineSurfaceTensionLut(float dist)
{
	static float factor = 1.0 / fc_->globalSupportRadius;
	int index = dist * factor * kLutSize;

	return (index >= kLutSize) ? 0.0 : lut_spline_surface_tension_[index];
}

float WeightingKernel::kernelSplineSurfaceAdhesion(float dist, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	/****************************************************************
				0.007 / h^3.25 * (-4.0*r^2/h + 6.0*r - 2.0*h)^1/4	,  h/2 < r <= h
	A(r) =		
				0, otherwise
	****************************************************************/

	const float tempConst = 0.007f/(powf(sr, 3.25));

	if (dist > 0.5 * sr && dist <= sr)
	{
		const float baseValue = -4.0f*dist*dist/sr + 6.0f*dist - 2.0f*sr;
		return tempConst * powf( baseValue, 1.0/4.0);
	} 
	else
	{
		return 0.0f;
	}
}

float WeightingKernel::kernelSplineSurfaceAdhesionLut(float dist)
{
	static float factor = 1.0 / fc_->globalSupportRadius;
	int index = dist * factor * kLutSize;

	return (index >= kLutSize) ? 0.0 : lut_spline_surface_adhesion_[index];
}

void WeightingKernel::kernelSurfaceTensionOldMethod(float distSq, float& grad, float &lap, float &clapSelf, float sr)
{
	// NOTE: This method assumes that |rvec| < fc_->support_radius !!
	// using the poly6 spline kernel for the color field:
	// W(r,h) = 315/(64*pi*h^9)*(h^2 - |r|^2)^3;
	// grad W = -945/(32*pi*h^9)*(h^2 - |r|^2)^2 * r   (?)
	// lap W  = -945/(32*pi*h^9)*(|r|^2-h^2)*(7*r^2 - 3*h^2)   (?)

	// grad is set to the factor that r has to be multiplied to get the gradient

	// calculate gradient
	//float r2mh2 = distSq - support_radius_sq_;
	if(distSq > sr*sr)
	{
		grad = 0.0f;
		lap = 0.0f;
		return;
	}
	kernel_surface_tension_const_ = - 945.f/(32.f *float(kPi)*pow(sr,9));
	clapSelf = -945.f*3.f/(32.f*float(kPi)*pow(sr,5));	

	float r2mh2 = distSq - sr*sr;
	float tmp = kernel_surface_tension_const_ * r2mh2;
	grad = tmp * r2mh2;

	// calculate laplacian
	//lap = tmp * (7*distSq - 3*support_radius_sq_);
	lap = tmp * (7*distSq - 3*sr*sr);
}

void WeightingKernel::initLookupTables()
{  
	float sr = fc_->globalSupportRadius; 
	for (int i = 0; i < kLutSize; i++) {
		float dist = sr * i / kLutSize;
		lut_kernel_m4_[i] = KernelM4(dist, sr);
		lut_kernel_pressure_grad_[i] = kernelPressureGrad(dist, sr);
		lut_kernel_viscosity_lap_[i] = kernelViscosityLaplacian(dist, sr);
		lut_spline_surface_tension_[i] = kernelSplineSurfaceTension(dist, sr);
		lut_spline_surface_adhesion_[i] = kernelSplineSurfaceAdhesion(dist, sr);
	}	

	lut_kernel_m4_[kLutSize] = 0.0;
	lut_kernel_pressure_grad_[kLutSize] = 0.0;
	lut_kernel_viscosity_lap_[kLutSize] = 0.0;
	lut_spline_surface_tension_[kLutSize] = 0.0;
	lut_spline_surface_adhesion_[kLutSize] = 0.0;
}

void WeightingKernel::calculateKernelConstants()
{
	/*
	float h = fc_->supportRadius;
	kernel_density_const_ = 315.0f/(64.0f * (float)kPi * h*h*h*h*h*h*h*h*h);	
	kernel_pressure_const_ = -45.f/((float(kPi)*h*h*h*h*h*h));
	kernel_viscosity_const_ = 45.f/((float(kPi)*h*h*h*h*h*h));
	kernel_surface_tension_const_ = - 945.f/(32.f *float(kPi)*pow(h,9));

	clapSelf = -945.f*3.f/(32.f*float(kPi)*pow(h,5));

	support_radius_sq_ = h*h;
	*/
}
