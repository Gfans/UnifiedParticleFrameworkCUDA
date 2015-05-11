#ifndef GPU_UNIFIED_RIGID_BODY_H_
#define GPU_UNIFIED_RIGID_BODY_H_ 

#include "UnifiedParticle.h"
#include "Matrix3x3.h"
#include "global.h"

#include "vmmlib/vector3.h"

// Represents a rigid body consisting of particles
class RigidBody
{
public:

	RigidBody();

	/**
	* Adds force to the sum of forces that will influence linear velocity 
	* (Force, not force density)
	*/
	inline void ApplyForce(const vmml::Vector3f& force);

	/**
	* Increases net torque that will cause angular momentum by applying 
	* force at the position of bodyPart (Force, not force density)
	*/
	void ApplyTorque(const vmml::Vector3f& force, const UnifiedParticle& body_part);

	/**
	* Inverts inertia if it's possible. Otherwise, invertedInertia is set 
	* to 3/trace(inertia) * Identity Matrix
	*/
	void InvertInertia();

	/*
	* Temperately add new particle index into the rigid_particle_buffers_
	*/
	inline void AddToParentRigidBody(const int i);

	/*
	* Clear rigid_particle_buffers_ 
	*/
	inline void ClearBuffers();

	/*
	* Update rigid_particle_indices_ 
	*/
	void UpdateRigidBodyParticleIndices();

	// TODO: The CPU version is not fully correct, should be fixed if the original one doesn't behave well in some cases
	void PerformLinearStep(const float delta);
	void PerformAngularStep(const float delta);
	void UpdateRotationMatrix();
	void UpdateInverseInertiaTensor();
	void PerformStep(const float delta);
	void UpdateMomenta(const float deltaT, const float terminalMomentum);
	//---------------------------------------------------------------------------------------------------

	std::vector<int>&			rigid_particle_indices()		{ return rigid_particle_indices_; }
	const std::vector<int>&		rigid_particle_indices() const  { return rigid_particle_indices_; }

	float mass() const { return mass_; }	
	void  set_mass (const float mass) { mass_ = mass; }

	vmml::Vector3f rigidbody_pos() const { return rigidbody_pos_; }
	void set_rigidbody_pos(const vmml::Vector3f& pos) { rigidbody_pos_ = pos; }

	vmml::Vector3f old_center_of_mass() const { return old_center_of_mass_; }
	void set_center_of_mass(const vmml::Vector3f& com) { old_center_of_mass_ = com; }

	vmml::Vector3f velocity() const { return velocity_; }
	void set_velocity(const vmml::Vector3f& vel) { velocity_ = vel; }

	vmml::Vector3f linear_momentum() const { return linear_momentum_; }
	void set_linear_momentum(const vmml::Vector3f& linear_momentum) { linear_momentum_ = linear_momentum; }

	vmml::Vector3f angular_velocity() const { return angular_velocity_; }
	void set_angular_velocity(const vmml::Vector3f& angular_velocity) { angular_velocity_ = angular_velocity; }

	vmml::Vector3f angular_momentum() const { return angular_momentum_; }
	void set_angular_momentum(const vmml::Vector3f& angular_momentum) { angular_momentum_ = angular_momentum; }

	Matrix3x3 inertia() const { return inertia_; }
	void set_inertia(const Matrix3x3& mat) { inertia_ = mat; }

	Matrix3x3 inverted_inertia() const { return inverted_inertia_; }
	void set_inverted_inertia(const Matrix3x3& mat) { inverted_inertia_ = mat; }

	Matrix3x3 inverted_inertia_local() const { return inverted_inertia_local_; }

	vmml::Vector4f quaternion() const { return quaternion_; }
	void set_quaternion(const vmml::Vector4f& vec) { quaternion_ = vec; }

	vmml::Vector3f torque() const { return torque_; }
	void set_torque(const vmml::Vector3f& vec) { torque_ = vec; }

	vmml::Vector3f force() const { return force_; }
	void set_force(const vmml::Vector3f& vec) { force_ = vec; }

	Matrix3x3 rotation_matrix() const { return rotation_matrix_; }
	void set_rotation_matrix(const Matrix3x3& mat) { rotation_matrix_ = mat; }

private:

	float mass_;									// sum of mass of all particles
	vmml::Vector3f rigidbody_pos_;	
	vmml::Vector3f old_center_of_mass_;				// center of mass before the last integration step (for use in UnifiedPhysics::timeIntegration())
	vmml::Vector3f velocity_;						// linear velocity = velocity at rigidbody_pos_
	vmml::Vector3f linear_momentum_;				// linear_momentum_ = mass * linear velocity

	/**
	* angular velocity "omega", where omega/|omega| is the axis of rotation
	* and |omega| is the amount of rotation
	*/
	vmml::Vector3f angular_velocity_;

	/**
	* angular_momentum_ = inertia * angular velocity
	* The angular momentum vector l is always measured for a particular frame of reference
	* here angular momentum about the center of mass of the body is used.
	*/
	vmml::Vector3f angular_momentum_;

	//------------------------ new methods : "Real-Time Rigid Body Interactions" ------------------------
	// TODO: it's not correct yet, should be fixed if the original one doesn't behave well in some cases
	/**
	* quaternion = [s, V] represents a rotation of s radians about an axis defined by the vector V
	* it's more accurate than rotationMatrix in physics simulation 
	*/
	vmml::Vector4f quaternion_;

	//---------------------------------------------------------------------------------------------------
	vmml::Vector3f force_;
	vmml::Vector3f torque_;	

	/**
	* The rotation matrix holds the rotation for one step.
	* defining rotationMatrix to only constitute the last step of rotation renders storing the original i ˆr for every
	particle unnecessary as well.
	*/
	Matrix3x3 rotation_matrix_;

	/**
	* r_i = particles[i]->position - body->rigidbody_pos_
	*
	* 							|	r_i[1]^2+r_i[2]^2	-r_i[0]*r_i[1]		-r_i[0]*r_i[2]		|
	* inertia=	sum_{i}	m_i * 	|	-r_i[0]*r_i[1]		r_i[0]^2+r_i[2]^2	-r_i[1]*r_i[2]		|
	*							|	-r_i[0]*r_i[2]		-r_i[1]*r_i[2]		r_i[0]^2+r_i[1]^2	|
	* (Eq. 5.9) from paper "Simulation of Fluid-Solid Interaction"
	*/
	Matrix3x3 inertia_;
	Matrix3x3 inverted_inertia_;
	Matrix3x3 inverted_inertia_world_;
	Matrix3x3 inverted_inertia_local_;

	std::vector<int> rigid_particle_indices_;		// rigid body particle indices array
	std::vector<int> rigid_particle_buffers_;		// Used to temperately store new particle indices after sorting

};

//--------------------------------------------------------------------
inline void RigidBody::ApplyForce(const vmml::Vector3f& force)
{
	this->force_+=force;
}

//--------------------------------------------------------------------
inline void RigidBody::AddToParentRigidBody(const int i)
{
	rigid_particle_buffers_.push_back(i);
}

//--------------------------------------------------------------------
inline void RigidBody::ClearBuffers()
{
	rigid_particle_buffers_.clear();
}

#endif	// GPU_UNIFIED_RIGID_BODY_H_
