#ifndef VECTOR4D_H
#define VECTOR4D_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "scalar.h"
#include "vector3d.h"

class Vector4DMeshVersion{
  friend std::ostream &operator<<(std::ostream&, const Vector4DMeshVersion&);
  friend std::istream &operator>>(std::istream &input, Vector4DMeshVersion& v);
 private:
  scalar array[4];
 public:

  Vector4DMeshVersion();

  Vector4DMeshVersion(const Vector3DMeshVersion& v);
  Vector4DMeshVersion(const Vector3DMeshVersion& v, scalar s);

  Vector4DMeshVersion(scalar x, scalar y, scalar z,scalar w);  

  void setLocation(scalar x,scalar y,scalar z,scalar w);
  std::string toString() const;

  scalar X() const{
    return array[0];
  }
  
  scalar Y() const{
    return array[1];
  }
  
  scalar Z() const{
    return array[2];
  }

  scalar W() const{
    return array[3];
  }

  scalar operator[](int i) const{
    return array[i];
  }

  scalar & operator[](int i){
    return array[i];
  }

  /* Dot Product */
  scalar operator*(const Vector4DMeshVersion & v) const;
  
  /* Vector Addition */
  Vector4DMeshVersion operator+(const Vector4DMeshVersion & v) const;
  void operator+=(const Vector4DMeshVersion & v);

  /* Vector Subtraction */
  Vector4DMeshVersion operator-(const Vector4DMeshVersion & v) const;
  void operator-=(const Vector4DMeshVersion & v);

  /* Scalar Multiplication */
  Vector4DMeshVersion operator*(const scalar f) const;
  void operator*=(const scalar f);

  /* Scalar Division */
  Vector4DMeshVersion operator/(const scalar f) const;
  void operator/=(const scalar f);

  /* Euclidean Norm of this Vector */
  scalar norm() const{
    scalar x,y,z,w;
    x = array[0] * array[0];
    y = array[1] * array[1];
    z = array[2] * array[2];
    w = array[2] * array[3];
    return sqrtf(x + y + z + w);
  }

  scalar norm2() const{
    scalar x,y,z,w;
    x = array[0] * array[0];
    y = array[1] * array[1];
    z = array[2] * array[2];
    w = array[2] * array[3];
    return (x + y + z + w);
  }

  scalar * loadFromArray(scalar * a){
    memcpy(array,a,4 * sizeof(scalar));
    return &a[4];
  }
  
  scalar * writeToArray(scalar * a){
    memcpy(a,array,4 * sizeof(scalar));
    return &a[4];
  }


};


inline Vector4DMeshVersion operator-(const Vector4DMeshVersion &u){
  return Vector4DMeshVersion(-u[0], -u[1], -u[2], -u[3]); 
}


inline Vector4DMeshVersion cross(const Vector4DMeshVersion& a,const Vector4DMeshVersion& b,const Vector4DMeshVersion& c){

    scalar d1 = (b[2] * c[3]) - (b[3] * c[2]);
    scalar d2 = (b[1] * c[3]) - (b[3] * c[1]);
    scalar d3 = (b[1] * c[2]) - (b[2] * c[1]);
    scalar d4 = (b[0] * c[3]) - (b[3] * c[0]);
    scalar d5 = (b[0] * c[2]) - (b[2] * c[0]);
    scalar d6 = (b[0] * c[1]) - (b[1] * c[0]);

    return Vector4DMeshVersion(- a[1] * d1 + a[2] * d2 - a[3] * d3,
		      a[0] * d1 - a[2] * d4 + a[3] * d5,
		    - a[0] * d2 + a[1] * d4 - a[3] * d6,
		      a[0] * d3 - a[1] * d5 + a[2] * d6);
}

 
inline void normalize(Vector4DMeshVersion& v){
  scalar l = v.norm();
  v /= l;
}

inline Vector4DMeshVersion operator*(scalar f, const Vector4DMeshVersion &v){
  Vector4DMeshVersion a(f * v.X(), f * v.Y(), f * v.Z(), f * v.W());
  return a;
}
  
inline std::ostream &operator<<(std::ostream &output, const Vector4DMeshVersion& v){
  return output << v.X() << " " << v.Y() << " " << v.Z() << " " << v.W();
}

inline std::istream &operator>>(std::istream &input, Vector4DMeshVersion& v){
  input >> v[0];
  input >> v[1];
  input >> v[2];
  input >> v[3];

  return input;
}

#endif
