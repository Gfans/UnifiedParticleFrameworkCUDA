#ifndef GPU_UNIFIED_VECTOR3D_H
#define GPU_UNIFIED_VECTOR3D_H

#include <math.h>
#include "vmmlib/vector3.h"

//--------------------------------------------------------------------
class Vector3D
	//--------------------------------------------------------------------
{	
public:

	inline Vector3D();
	inline Vector3D(float v0, float v1, float v2);
	inline Vector3D(vmml::Vector3f v);
	inline ~Vector3D();

	inline void Set (float v0, float v1, float v2);
	inline void SetZero();
	inline void MakeNegative();
	inline float GetSquaredLength();
	inline float GetLength();

	inline float GetSquaredLengthXZ();
	inline float GetLengthXZ();
	/*
	float getPeriodicDistanceSq(float x1, float y1, float z1,
	float x2, float y2, float z2,
	float *dx, float *dy, float *dz);
	*/
	inline float Normalize ();
	static inline Vector3D CrossProduct(const Vector3D &a, const Vector3D &b);
	static inline float DotProduct(const Vector3D &a, const Vector3D &b);

	inline Vector3D& operator= (const Vector3D& V);
	inline Vector3D& operator+= (const Vector3D V);
	inline Vector3D& operator+= (float sum);
	inline Vector3D operator+ (const Vector3D V);
	inline Vector3D& operator-= (const Vector3D V);
	inline Vector3D& operator-= (float sub);
	inline Vector3D operator- (const Vector3D V);
	inline Vector3D operator- (float sub);
	inline Vector3D operator- ();
	inline Vector3D& operator*= (const Vector3D V);
	inline Vector3D& operator*= (float m);
	inline Vector3D operator* (const Vector3D V);
	inline Vector3D operator* (float m);
	inline Vector3D& operator/= (const Vector3D V);
	inline Vector3D& operator/= (float d);
	inline Vector3D operator/ (float d);
	inline bool operator == (const Vector3D &a);
	inline bool operator != (const Vector3D &a);
	inline float& operator[] (int index);	

public:
	float v[3];
};




//--------------------------------------------------------------------
Vector3D::Vector3D()
	//--------------------------------------------------------------------
{

}

//--------------------------------------------------------------------
Vector3D::Vector3D(vmml::Vector3f v)
	//--------------------------------------------------------------------
{
	v[0] = v.x;
	v[1] = v.y;
	v[2] = v.z;
}

//--------------------------------------------------------------------
Vector3D::Vector3D(float v0, float v1, float v2)
	//--------------------------------------------------------------------
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;
}

//--------------------------------------------------------------------
Vector3D::~Vector3D()
	//--------------------------------------------------------------------
{
}


//--------------------------------------------------------------------
inline void Vector3D::Set(float v0, float v1, float v2)
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;	
}


//--------------------------------------------------------------------
inline void Vector3D::SetZero()
{
	v[0] = 0;
	v[1] = 0;
	v[2] = 0;	
}

//--------------------------------------------------------------------
inline void Vector3D::MakeNegative()
{
	v[0] = -v[0];
	v[1] = -v[1];
	v[2] = -v[2];	
}

//--------------------------------------------------------------------
inline float Vector3D::GetSquaredLength()
{
	return(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

//--------------------------------------------------------------------
inline float Vector3D::GetLength()
{
	return (float)sqrt(GetSquaredLength());
}

//--------------------------------------------------------------------
inline float Vector3D::GetSquaredLengthXZ()
{
	return(v[0]*v[0] + v[2]*v[2]);
}

//--------------------------------------------------------------------
inline float Vector3D::GetLengthXZ()
{
	return (float)sqrt(GetSquaredLengthXZ());
}

/*
//--------------------------------------------------------------------
float Vector3D::getPeriodicDistanceSq(float x1, float y1, float z1,
float x2, float y2, float z2,
float *dx, float *dy, float *dz)
//return variables
//--------------------------------------------------------------------
{
}
*/

//--------------------------------------------------------------------
inline float Vector3D::Normalize()
{
	float length = GetLength();
	if (length == 0.0f)
		return 0;

	float rezLength = 1.0f / length;
	v[0] *= rezLength;
	v[1] *= rezLength;
	v[2] *= rezLength;
	return length;
}

//--------------------------------------------------------------------
inline Vector3D Vector3D::CrossProduct(const Vector3D &a, const Vector3D &b)
{
	Vector3D result;

	result.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
	result.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
	result.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];

	return(result);
}

//--------------------------------------------------------------------
inline float Vector3D::DotProduct(const Vector3D &a, const Vector3D &b)
{
	return(a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator= (const Vector3D& V)
	//--------------------------------------------------------------------
{
	v[0] = V.v[0];
	v[1] = V.v[1];
	v[2] = V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator+= (const Vector3D V)
	//--------------------------------------------------------------------
{
	v[0] += V.v[0];
	v[1] += V.v[1];
	v[2] += V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator+= (float sum)
	//--------------------------------------------------------------------
{
	v[0] += sum;
	v[1] += sum;
	v[2] += sum;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator+ (const Vector3D V)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] + V.v[0];
	res.v[1] = v[1] + V.v[1];
	res.v[2] = v[2] + V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator-= (const Vector3D V)
	//--------------------------------------------------------------------
{
	v[0] -= V.v[0];
	v[1] -= V.v[1];
	v[2] -= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator-= (float sub)
	//--------------------------------------------------------------------
{
	v[0] -= sub;
	v[1] -= sub;
	v[2] -= sub;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- (const Vector3D V)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] - V.v[0];
	res.v[1] = v[1] - V.v[1];
	res.v[2] = v[2] - V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- (float sub)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] - sub;
	res.v[1] = v[1] - sub;
	res.v[2] = v[2] - sub;
	return (res);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- ()
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = -v[0];
	res.v[1] = -v[1];
	res.v[2] = -v[2];
	return (res);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator*= (const Vector3D V)
	//--------------------------------------------------------------------
{
	v[0] *= V.v[0];
	v[1] *= V.v[1];
	v[2] *= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator*= (float m)
	//--------------------------------------------------------------------
{
	v[0] *= m;
	v[1] *= m;
	v[2] *= m;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator* (const Vector3D V)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] * V.v[0];
	res.v[1] = v[1] * V.v[1];
	res.v[2] = v[2] * V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator* (float m)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] * m;
	res.v[1] = v[1] * m;
	res.v[2] = v[2] * m;
	return (res); 
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator/= (const Vector3D V)
	//--------------------------------------------------------------------
{
	v[0] /= V.v[0];
	v[1] /= V.v[1];
	v[2] /= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator/= (float d)
	//--------------------------------------------------------------------
{
	v[0] /= d;
	v[1] /= d;
	v[2] /= d;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator/ (float d)
	//--------------------------------------------------------------------
{
	Vector3D result;
	result.v[0] = v[0] / d;
	result.v[1] = v[1] / d;
	result.v[2] = v[2] / d;
	return result;
}

//--------------------------------------------------------------------
bool Vector3D::operator == (const Vector3D &a)
	//--------------------------------------------------------------------
{
	return(v[0] == a.v[0] && v[1] == a.v[1] && v[2] == a.v[2]);
}

//--------------------------------------------------------------------
bool Vector3D::operator != (const Vector3D &a)
	//--------------------------------------------------------------------
{
	return(v[0] != a.v[0] || v[1] != a.v[1] || v[2] != a.v[2]);
}

//--------------------------------------------------------------------
float& Vector3D::operator[] (int index)
	//--------------------------------------------------------------------
{
	return (v[index]);
}


#endif // GPU_UNIFIED_VECTOR3D_H
