#ifndef MATRIX4D_H
#define MATRIX4D_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "vector4d.h"

class Matrix4DMeshVersion{
  friend std::ostream &operator<<(std::ostream&, const Matrix4DMeshVersion&);
 private:
  Vector4DMeshVersion row[4];
 public:
  /* Constructors */
  Matrix4DMeshVersion();
  Matrix4DMeshVersion(scalar x1y1, scalar x1y2, scalar x1y3, scalar x1y4,
	   scalar x2y1, scalar x2y2, scalar x2y3, scalar x2y4,
	   scalar x3y1, scalar x3y2, scalar x3y3, scalar x3y4,
	   scalar x4y1, scalar x4y2, scalar x4y3, scalar x4y4);  
  Matrix4DMeshVersion(const Vector4DMeshVersion& r0, const Vector4DMeshVersion& r1, const Vector4DMeshVersion& r2, const Vector4DMeshVersion& r3);

  static Matrix4DMeshVersion I();
  static Matrix4DMeshVersion Zero();

  std::string toString() const;
  
  /* Matrix Vector Product */
  Vector4DMeshVersion operator*(const Vector4DMeshVersion & v) const;
  
  /* Matrix Addition */
  Matrix4DMeshVersion  operator+ (const Matrix4DMeshVersion & m) const;
  Matrix4DMeshVersion& operator+=(const Matrix4DMeshVersion & m);
  
  /* Matrix Multiplication */
  Matrix4DMeshVersion operator*(const Matrix4DMeshVersion & m) const;

  /* Matrix Subtraction */
  Matrix4DMeshVersion  operator- (const Matrix4DMeshVersion & m) const;
  Matrix4DMeshVersion& operator-=(const Matrix4DMeshVersion & m);

  /* Scalar Multiplication */
  Matrix4DMeshVersion  operator* (scalar f) const;
  Matrix4DMeshVersion& operator*=(scalar f);

  /* Scalar Division */
  Matrix4DMeshVersion  operator/(scalar f) const;
  Matrix4DMeshVersion& operator/=(scalar f);

  /* The Trace of this Matrix */
  scalar trace() const;

  Matrix4DMeshVersion transpose() const;

  scalar det() const;

  Matrix4DMeshVersion adjoint() const;
  Matrix4DMeshVersion inverse() const;

  /* Accessors */
  Vector4DMeshVersion&       operator[](int i)       { return row[i]; }
  const Vector4DMeshVersion& operator[](int i) const { return row[i]; }

  scalar& operator()(int i, int j)       { return row[i][j]; }
  scalar  operator()(int i, int j) const { return row[i][j]; }

  Vector4DMeshVersion col(int j) const {return Vector4DMeshVersion(row[0][j],row[1][j],row[2][j],row[3][j]);}
};


inline void transpose(Matrix4DMeshVersion & m){
  scalar temp;
  for(int i = 0; i < 4; i++){
    for(int j = i+1; j < 4; j++){
      temp = m[i][j];
      m[i][j] = m[j][i];
      m[j][i] = temp;
    }
  }
}
    
	
inline Matrix4DMeshVersion outer(const Vector4DMeshVersion &u, const Vector4DMeshVersion &v){
  return Matrix4DMeshVersion(u.X() * v.X(), u.X() * v.Y(), u.X() * v.Z(), u.X() * v.W(),
		  u.Y() * v.X(), u.Y() * v.Y(), u.Y() * v.Z(), u.Y() * v.W(),
		  u.Z() * v.X(), u.Z() * v.Y(), u.Z() * v.Z(), u.Z() * v.W(),
		  u.W() * v.X(), u.W() * v.Y(), u.W() * v.Z(), u.W() * v.W());
}




inline Matrix4DMeshVersion operator*(scalar f, const Matrix4DMeshVersion &m){
  return m*f;
}
  
inline std::ostream &operator<<(std::ostream &output, const Matrix4DMeshVersion& m){
  return output << m.toString();
}

#endif
