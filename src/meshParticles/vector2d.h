#ifndef VECTOR2D_H
#define VECTOR2D_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "scalar.h"


class Vector2DMeshVersion{
  friend std::ostream &operator<<(std::ostream&, const Vector2DMeshVersion&);
  friend std::istream &operator>>(std::istream &input, Vector2DMeshVersion& v);
 private:
  scalar array[2];
 public:

  Vector2DMeshVersion(){
    array[0] = 0;
    array[1] = 0;
  }

  Vector2DMeshVersion(scalar a){
    array[0] = a;
    array[1] = a;
  }

  Vector2DMeshVersion(scalar x, scalar y){
    array[0] = x;
    array[1] = y;
  }

  static unsigned int Dimension() { return 2; }

  /* Access Specific entries */
  scalar X() const{ return array[0]; }
  scalar Y() const{ return array[1]; }

  scalar  operator[](int i) const{ return array[i]; }
  scalar& operator[](int i)      { return array[i]; }

  /* Equality operators */
  bool operator==(const Vector2DMeshVersion & v) const{  return (X() == v.X()) && (Y() == v.Y()); }
  bool operator!=(const Vector2DMeshVersion & v) const{  return (X() != v.X()) || (Y() != v.Y()); }

  /* Dot Product */
  scalar operator*(const Vector2DMeshVersion & v) const {  return array[0] * v.X() + array[1] * v.Y(); }
  
  /* Vector Addition */
  Vector2DMeshVersion operator+(const Vector2DMeshVersion & v) const{  return Vector2DMeshVersion(array[0] + v.X(),array[1] + v.Y());}
  void operator+=(const Vector2DMeshVersion & v) {   array[0] += v.X();  array[1] += v.Y(); }

  /* Vector Subtraction */
  Vector2DMeshVersion operator-(const Vector2DMeshVersion & v) const{  return Vector2DMeshVersion( X() - v.X(), Y() - v.Y());}
  void operator-=(const Vector2DMeshVersion & v) { array[0] -= v.X(); array[1] -= v.Y(); }

  /* Scalar Multiplication */
  Vector2DMeshVersion operator*(const scalar f) const { return Vector2DMeshVersion(f*X(),f*Y()); }
  void operator*=(const scalar f) { array[0] *= f; array[1] *= f; }

  /* Scalar Division */
  Vector2DMeshVersion operator/(const scalar f) const{
    scalar f_inv = ((scalar) 1.0)/f;  
    return Vector2DMeshVersion(f_inv * X(), f_inv * Y());
  }

  void operator/=(const scalar f){
    scalar f_inv = ((scalar)1.0)/f;  
    array[0] *= f_inv;
    array[1] *= f_inv;
  }

  /* Euclidean Norm of this Vector */
  scalar norm() const{ return SQRT(X() * X() + Y() * Y()); }

  /* Square of the Norm of this Vector */
  scalar norm2() const{ return (X() * X() + Y() * Y()); }
   
  /* Return a normalized copy of this Vector */
  Vector2DMeshVersion normalized() const{ return (*this) / norm(); }
 
  scalar * loadFromArray(scalar * a){
    memcpy(array,a,Dimension() * sizeof(scalar));
    return &a[Dimension()];
  }

  scalar * writeToArray(scalar * a){
    memcpy(a,array,Dimension() * sizeof(scalar));
    return &a[Dimension()];
  }


  std::string toString() const{
    std::ostringstream s;
    s << "(" << X() << "," << Y() << ")";
    return s.str();
  } 
};


inline void normalize(Vector2DMeshVersion& v){
  v /= v.norm();
}

inline Vector2DMeshVersion operator-(const Vector2DMeshVersion &u){
  return Vector2DMeshVersion(-u.X(), -u.Y());
}


inline Vector2DMeshVersion operator*(scalar f, const Vector2DMeshVersion &v){
  return Vector2DMeshVersion(f * v.X(), f * v.Y());
}
  
inline std::ostream &operator<<(std::ostream &output, const Vector2DMeshVersion& v){
  return output << v.X() << " " << v.Y();
}

inline std::istream &operator>>(std::istream &input, Vector2DMeshVersion& v){
  input >> v[0];
  input >> v[1];

  return input;
}

#endif
