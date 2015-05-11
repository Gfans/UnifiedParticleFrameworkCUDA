#include "RigidBody.h"

//--------------------------------------------------------------------
RigidBody::RigidBody(): mass_(0.0), rigidbody_pos_(0.0f, 0.0f ,0.0f), old_center_of_mass_(0.0f, 0.0f ,0.0f), velocity_(0.0f, 0.0f ,0.0f), linear_momentum_(0.0f, 0.0f ,0.0f),
	angular_velocity_(0.0f, 0.0f ,0.0f), angular_momentum_(0.0f, 0.0f ,0.0f), quaternion_(1.0f, 0.0f, 0.0f, 0.0f), force_(0.0f, 0.0f ,0.0f), torque_(0.0f, 0.0f ,0.0f)
	//--------------------------------------------------------------------
{
	inertia_.SetZero();
	inverted_inertia_.SetZero();		
	inverted_inertia_world_.SetZero();
	inverted_inertia_local_.SetZero();	
	rotation_matrix_=Matrix3x3::UnitMatrix(); 
}

//--------------------------------------------------------------------
void RigidBody::ApplyTorque(const vmml::Vector3f& force, const UnifiedParticle& body_part)
{
	torque_ += (vmml::Vector3f(body_part.position) - rigidbody_pos_).cross(force);
}

//--------------------------------------------------------------------
void RigidBody::InvertInertia()
{
	bool success = Matrix3x3::CalculateInverse(inertia_, inverted_inertia_);
	if(!success)
	{
		inverted_inertia_ = Matrix3x3::UnitMatrix()*(3/(inertia_.elements[0][0]+inertia_.elements[1][1]+inertia_.elements[2][2]));
	}
}

//--------------------------------------------------------------------
void RigidBody::UpdateRigidBodyParticleIndices()
{
	// clear old values
	rigid_particle_indices_.clear();

	// copy the new ones from buffers
	rigid_particle_indices_ = rigid_particle_buffers_;

	// Note: we don't clear buffers here cause we do it every time before we change the particles' parent rigid body information right after sorting

}

//------------------------ new methods : "Real-Time Rigid Body Interactions" ------------------------

//--------------------------------------------------------------------
void RigidBody::PerformLinearStep(const float delta)
{
	// Equation 2 : v = P/M & Equation 3 : dx/dt = v
	velocity_ = linear_momentum_ / mass_;
	rigidbody_pos_ += velocity_ * delta;
}

//--------------------------------------------------------------------
void RigidBody::PerformAngularStep(const float delta)
{
	// Equation 5 : w = I^-1 * L
	// update angular velocity
	angular_velocity_ = inverted_inertia_world_ * angular_momentum_;
	float angular_speed = angular_velocity_.length();
	if (angular_speed > 0.0f)
	{
		// rotation axis a = w/|w| 
		vmml::Vector3f rotation_axis = angular_velocity_.getNormalized();

		// theta = |w * dt|
		float rotation_angle = angular_speed*delta;

		//TODO: use advanced quaternion lib functions
		// dq = (cos(theta/2), a * sin(theta/2))
		float ds = cos(rotation_angle/2.0f);
		float dvx = rotation_axis[0]*sin(rotation_angle/2.0f);
		float dvy = rotation_axis[1]*sin(rotation_angle/2.0f);
		float dvz = rotation_axis[2]*sin(rotation_angle/2.0f);
		vmml::Vector4f dq(ds, dvx, dvy, dvz);

		// q(t)
		float s = quaternion_[0];
		float vx = quaternion_[1];
		float vy = quaternion_[2];
		float vz = quaternion_[3];

		// q(t+dt) = dq x q(t)
		quaternion_[0] = s*ds - vx*dvx - vy*dvy - vz*dvz;
		quaternion_[1] = ds*vx + s*dvx + dvy*vz - dvz*vy;
		quaternion_[2] = ds*vy + s*dvy + dvz*vx - dvx*vz;
		quaternion_[3] = ds*vz + s*dvz + dvx*vy - dvy*vx;
	}	
}

//--------------------------------------------------------------------
void RigidBody::UpdateRotationMatrix()
{
	// first normalize quaternion
	quaternion_.getNormalized();

	// then calculate rotation matrix from quaternion q = [s, v_x, v_y, v_z] 
	/************************************************************************/
	/*  1-2*v_y*v_y-2*v_z*v_z  2*v_x*v_y-2*s*v_z       2*v_x*v_z+2*s*v_y    */
	/*  2*v_x*v_y+2*s*v_z      1-2*v_x*v_x-2*v_z*v_z   2*v_y*v_z-2*s*v_x    */
	/*  2*v_x*v_z-2*s*v_y      2*v_y*v_z+2*s*v_x       1-2*v_x*v_x-2*v_y*v_y*/
	/************************************************************************/

	float w = quaternion_[0];
	float x = quaternion_[1];
	float y = quaternion_[2];
	float z = quaternion_[3];

	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float xy = x * y;
	float xz = x * z;
	float yz = y * z;
	float wx = w * x;
	float wy = w * y;
	float wz = w * z; 

	rotation_matrix_.elements[0][0] = 1.0f-2.0f*(yy+zz);
	rotation_matrix_.elements[0][1] = 2.0f*(xy-wz);
	rotation_matrix_.elements[0][2] = 2.0f*(xz+wy);
	rotation_matrix_.elements[1][0] = 2.0f*(xy+wz);
	rotation_matrix_.elements[1][1] = 1.0f-2.0f*(xx+zz);
	rotation_matrix_.elements[1][2] = 2.0f*(yz-wx);
	rotation_matrix_.elements[2][0] = 2.0f*(xz-wy);
	rotation_matrix_.elements[2][1] = 2.0f*(yz+wx);
	rotation_matrix_.elements[2][2] = 1.0f-2.0f*(xx+yy);
}

//--------------------------------------------------------------------
void RigidBody::UpdateInverseInertiaTensor()
{
	// Equation 6 : I^-1_world = R * I^-1_local * R^T  
	inverted_inertia_world_ = rotation_matrix_ * inverted_inertia_local_ * rotation_matrix_.GetTransposedMatrix();
}

//--------------------------------------------------------------------
void RigidBody::PerformStep(const float delta)
{
	UpdateInverseInertiaTensor();
	PerformLinearStep(delta);
	PerformAngularStep(delta);
}

//--------------------------------------------------------------------
void RigidBody::UpdateMomenta(const float deltaT, const float terminalMomentum)
{
	// Equation 1 : dP/dt = F & Equation 4 : dL/dt = r x F
	linear_momentum_ += force_ * deltaT;
	// clamp operation
	for (int i = 0; i < 3; ++i)
	{
		if (linear_momentum_[i] > 0.0f)
			linear_momentum_[i] = MIN(linear_momentum_[i],terminalMomentum);	
		else
			linear_momentum_[i] = MAX(linear_momentum_[i],-terminalMomentum);
	}

	angular_momentum_ += torque_ * deltaT;
}

//------------------------------------------------------------------------------------------------------------------------