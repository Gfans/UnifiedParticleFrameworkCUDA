#include <iostream>
#include <limits>
#include "UnifiedParticle.h"

//--------------------------------------------------------------------
UnifiedParticle::UnifiedParticle()
	//--------------------------------------------------------------------
{

}

//--------------------------------------------------------------------
UnifiedParticle::UnifiedParticle(float x, float y, float z, double rest_density, double particle_mass, int type)
	//--------------------------------------------------------------------
{
	position_.set(x, y, z);
	predicted_position_ = position_;
	velocity_.set(0.0f, 0.0f, 0.0f);
	predicted_velocity_.set(0.0f, 0.0f, 0.0f);
	velocity_leapfrog_.set(0.0f, 0.0f, 0.0f);
	rest_density_ = rest_density;
	particle_mass_ = particle_mass;
	type_ = type;
	weighted_volume_ = particle_mass_ / rest_density_;
	shepard_filter_den_kernel_factor_ = 0.0f;
	SetParticleParameters();
}

//--------------------------------------------------------------------
UnifiedParticle::UnifiedParticle(const vmml::Vector3f& pos, const int type, const vmml::Vector3f& vel, const float corr_pressure, const float predicted_density, const float rest_density, double particle_mass)
//--------------------------------------------------------------------
{
	position_ = pos;
	type_ = type;
	velocity_ = vel;
	correction_pressure_ = corr_pressure;
	predicted_density_ = predicted_density;
	rest_density_ = rest_density;
	particle_mass_ = particle_mass;

	predicted_position_ = pos;
	predicted_velocity_ = vel;
	weighted_volume_ = particle_mass_ / rest_density;
	shepard_filter_den_kernel_factor_ = 0.0f;
	SetParticleParameters();
}

//--------------------------------------------------------------------
UnifiedParticle::~UnifiedParticle()
//--------------------------------------------------------------------
{

}

//--------------------------------------------------------------------
void UnifiedParticle::SetParticleParameters()
//--------------------------------------------------------------------
{
	init_relative_pos_.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	force_.set(0.0f, 0.0f, 0.0f);
	normalized_normal_.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	gas_const_ = 10;

	temperature_ = 0;

	// elastic/rigid body properties
	poisson_ratio_ = 0.3f; // range about: -0.8 <= poissonRatio <=0.45
	memset(plastic_strain_, 0, 6*sizeof(float)); // no plasticStrain in the beginning
	volume_conserving_coeff_ = 0.0f;// 100000000.0f
	
	elastic_limit_ = 1000.0f; // 1000.0f; // 0.0044f;
	plastic_limit_ = 0.0f;    // 0.0f;    // 0.486f;
	
	old_temperature_ = temperature_;
	solid_properties_.temperature = 20.0f;
	young_modulus_ = solid_properties_.young_modulus = 20000; // 2-particle-test: stable up to about 20000
	visc_const_ = solid_properties_.visc_const = 20.0f;//20.0f;
	fluid_properties_.temperature = 50.0f;
	fluid_properties_.young_modulus = 0.0f;
	fluid_properties_.visc_const = 3.0f;
	
	parent_rigidbody_ = NULL; // by default, particles are not parts of a rigid body
	rigidbody_ = false;
	has_made_solid_ = false;

	parent_rigid_body_index_ = -1;			// not a useful rigid body index
	order_in_child_particles_array_ = -1;		// not a useful index
}

//--------------------------------------------------------------------
float UnifiedParticle::GetPositionX()
{
	return(position_[0]);
}

//--------------------------------------------------------------------
float UnifiedParticle::GetPositionY()
//--------------------------------------------------------------------
{
	return(position_[1]);
}

//--------------------------------------------------------------------
float UnifiedParticle::GetPositionZ()
//--------------------------------------------------------------------
{
	return(position_[2]);
}

//--------------------------------------------------------------------
void UnifiedParticle::CalculateTempColor()
//--------------------------------------------------------------------
{
	if((temperature_ <= 33.0) && (temperature_ >= 0.0))
	{
		float tmp = temperature_/33.0;	
		color_.set(0.0, 0.0, tmp);		
	}
	else if((temperature_ <= 66.0) && (temperature_ > 33.0))
	{
		float tmp = (temperature_-33.0)/(66.0-33.0);	
		color_.set(tmp, 0.0, 1-tmp);	
	}
	else if((temperature_ <= 100.0) && (temperature_ > 66.0))
	{
		float tmp = (temperature_-66.0)/(100.0-66.0);
		color_.set(1.0, tmp, 0.0);
	}	
	else
	{
		color_.set(1.0, 1.0, 1.0);
	}
}

//--------------------------------------------------------------------
void UnifiedParticle::CalculateHookeMatrix()
//--------------------------------------------------------------------
{
	const float factor = young_modulus_/((1+poisson_ratio_)*(1-2*poisson_ratio_));
	/*
						1-poissonRatio	poissonRatio	poissonRatio	0					0					0
						poissonRatio	1-poissonRatio	poissonRatio	0					0					0
	hookeMatrix=factor*	poissonRatio	poissonRatio	1-poissonRatio	0					0					0
						0				0				0				1-2*poissonRatio	0					0
						0				0				0				0					1-2*poissonRatio	0
						0				0				0				0					0					1-2*poissonRatio
	
	*/
	// initialise with 0
	memset(hooke_matrix_, 0, 36*sizeof(float));
	
	// diagonal
	float diag=factor*(1-poisson_ratio_);
	float factor_times_pr=factor*poisson_ratio_;
	hooke_matrix_[0][0]=hooke_matrix_[1][1]=hooke_matrix_[2][2]=diag;
	hooke_matrix_[3][3]=hooke_matrix_[4][4]=hooke_matrix_[5][5]=diag-factor_times_pr;

	// 0,0 to 3,3 except diagonal
	for(int i=0; i<3; ++i)
	{
		for(int j=0; j<3; ++j)
		{
			if(i!=j)
			{
				hooke_matrix_[i][j]=factor_times_pr;
			}
		}
	}
}

//--------------------------------------------------------------------
UnifiedParticle::Properties UnifiedParticle::Properties::operator+(const UnifiedParticle::Properties& summand)
//--------------------------------------------------------------------
{
	Properties result;
	result.young_modulus = young_modulus + summand.young_modulus;
	result.visc_const = visc_const + summand.visc_const;
	return result;
}
	
//--------------------------------------------------------------------
UnifiedParticle::Properties UnifiedParticle::Properties::operator-(const UnifiedParticle::Properties& subtrahend)
//--------------------------------------------------------------------
{
	Properties result;
	result.young_modulus = young_modulus - subtrahend.young_modulus;
	result.visc_const = visc_const - subtrahend.visc_const;
	return result;
}

//--------------------------------------------------------------------
UnifiedParticle::Properties UnifiedParticle::Properties::operator*(float factor)
//--------------------------------------------------------------------
{
	Properties result;
	result.young_modulus = young_modulus * factor;
	result.visc_const = visc_const * factor;
	return result;
}
