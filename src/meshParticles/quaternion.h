#ifndef QUATERNION_H
#define QUATERNION_H

#include "vector3d.h"

#include "matrix3d.h"
#include "matrix4d.h"

/* 
 * Quaternion class, adapted from libgfx source code 
 */

class Quaternion{
 private:
  scalar s;
  Vector3DMeshVersion v;

 public:

  Quaternion()                                       { v = Vector3DMeshVersion(0.0, 0.0, 0.0); s=1.0; }
  Quaternion(scalar x, scalar y, scalar z, scalar w) { v[0]=x;v[1]=y;v[2]=z; s=w; }
  Quaternion(const Vector3DMeshVersion& a, scalar b)            { v=a; s=b; }
  Quaternion(const Quaternion& q)                    { *this=q; }
  
  // Access methods
  const Vector3DMeshVersion& V() const { return v; }
  Vector3DMeshVersion&       V()       { return v; }
  scalar      S() const { return s; }
  scalar&     S()       { return s; }
  
  // Assignment and in-place arithmetic methods
  Quaternion& operator=(const Quaternion& q);
  Quaternion& operator+=(const Quaternion& q);
  Quaternion& operator-=(const Quaternion& q);
  Quaternion& operator=(scalar d);
  Quaternion& operator*=(scalar d);
  Quaternion& operator/=(scalar d);
  
  Matrix3DMeshVersion toMatrix3d() const;
  Matrix4DMeshVersion toMatrix4d() const;

  Matrix3DMeshVersion unitToMatrix3d() const;
  Matrix4DMeshVersion unitToMatrix4d() const;

  static Quaternion RandomUnitQuaternion();


  scalar norm()  const{ return SQRT(s * s + v.norm2()); }
  scalar norm2() const{ return      s * s + v.norm2();  }

  scalar * loadFromArray(scalar * a){
    v.loadFromArray(a);
    s = a[3];
    return &a[4];
  }
  
  scalar * writeToArray(scalar * a){
    v.writeToArray(a);
    a[3] = s;
    return &a[4];
  }



  // Construction of standard quaternions
  static Quaternion ident();
};


////////////////////////////////////////////////////////////////////////
//
// Implementation of Quaternion methods
//

inline Quaternion& Quaternion::operator=(const Quaternion& q) { v=q.v; s=q.s; return *this; }
inline Quaternion& Quaternion::operator+=(const Quaternion& q) { v+=q.v; s+=q.s; return *this; }
inline Quaternion& Quaternion::operator-=(const Quaternion& q) { v-=q.v; s-=q.s; return *this; }

inline Quaternion& Quaternion::operator=(scalar d)  { v = Vector3DMeshVersion(d,d,d);  s=d;  return *this; }
inline Quaternion& Quaternion::operator*=(scalar d) { v*=d; s*=d; return *this; }
inline Quaternion& Quaternion::operator/=(scalar d) { v/=d; s/=d; return *this; }

inline Quaternion Quaternion::ident() { return Quaternion(0, 0, 0, 1); }

////////////////////////////////////////////////////////////////////////
//
// Standard arithmetic operators on quaternions
//

inline Quaternion operator+(const Quaternion& q, const Quaternion& r)
	{ return Quaternion(q.V()+r.V(), q.S()+r.S()); }

inline Quaternion operator*(const Quaternion& q, const Quaternion& r)
{
    return Quaternion(cross(q.V(),r.V()) +
		r.S()*q.V() +
		q.S()*r.V(),
		q.S()*r.S() - q.V()*r.V());
}

inline Quaternion operator*(const Quaternion& q, scalar s)
	{ return Quaternion(q.V()*s, q.S()*s); }
inline Quaternion operator*(scalar s, const Quaternion& q)
	{ return Quaternion(q.V()*s, q.S()*s); }

inline Quaternion operator/(const Quaternion& q, scalar s)
	{ return Quaternion(q.V()/s, q.S()/s); }

inline std::ostream &operator<<(std::ostream &out, const Quaternion& q)
	{ return out << q.V() << " " << q.S(); }

inline std::istream &operator>>(std::istream &in, Quaternion& q)
	{ return in >> q.V() >> q.S(); }


////////////////////////////////////////////////////////////////////////
//
// Standard functions on quaternions
//

inline scalar norm     (const Quaternion& q){ return SQRT(q.S()*q.S() + q.V().norm2()); }
inline scalar norm2    (const Quaternion& q){ return      q.S()*q.S() + q.V().norm2();  }
inline void   normalize(Quaternion& q)      { q /= norm(q); }


inline Quaternion conjugate(const Quaternion& q) { return Quaternion(-q.V(), q.S()); }
inline Quaternion inverse(const Quaternion& q)   { return conjugate(q)/norm2(q); }
inline void unitize(Quaternion& q)  { q /= norm(q); }

extern Vector3DMeshVersion rotation_axis (const Quaternion& q);
extern scalar   rotation_angle(const Quaternion& q);

extern Quaternion exp(const Quaternion& q);
extern Quaternion log(const Quaternion& q);
extern Quaternion axis_to_quaternion(const Vector3DMeshVersion& a, scalar phi);
extern Quaternion slerp(const Quaternion& from, const Quaternion& to, scalar t);

#endif
