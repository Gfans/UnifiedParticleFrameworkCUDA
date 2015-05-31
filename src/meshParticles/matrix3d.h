#ifndef MATRIX3D_H
#define MATRIX3D_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "vector3d.h"

class Matrix3DMeshVersion{
  friend std::ostream &operator<<(std::ostream&, const Matrix3DMeshVersion&);
  friend std::istream &operator>>(std::istream &input, Matrix3DMeshVersion& v);
 private:
  Vector3DMeshVersion row[3];
 public:
  /* Constructors */
  Matrix3DMeshVersion(){}
  Matrix3DMeshVersion(scalar x1y1, scalar x1y2, scalar x1y3,
	   scalar x2y1, scalar x2y2, scalar x2y3,
	   scalar x3y1, scalar x3y2, scalar x3y3);  
  Matrix3DMeshVersion(const Vector3DMeshVersion& r0, const Vector3DMeshVersion& r1, const Vector3DMeshVersion& r2);

  static Matrix3DMeshVersion I();
  static Matrix3DMeshVersion Zero();

  /* Matrix Vector Product */
  Vector3DMeshVersion operator*(const Vector3DMeshVersion & v) const;
  
  /* Matrix Addition */
  Matrix3DMeshVersion  operator+ (const Matrix3DMeshVersion & m) const;
  Matrix3DMeshVersion& operator+=(const Matrix3DMeshVersion & m);
  
  /* Matrix Multiplication */
  Matrix3DMeshVersion operator*(const Matrix3DMeshVersion & m) const;

  /* Matrix Subtraction */
  Matrix3DMeshVersion  operator- (const Matrix3DMeshVersion & m) const;
  Matrix3DMeshVersion& operator-=(const Matrix3DMeshVersion & m);

  /* Scalar Multiplication */
  Matrix3DMeshVersion  operator* (scalar f) const;
  Matrix3DMeshVersion& operator*=(scalar f);

  /* Scalar Division */
  Matrix3DMeshVersion  operator/(scalar f) const;
  Matrix3DMeshVersion& operator/=(scalar f);

  /* The Trace of this Matrix */
  scalar trace() const;

  Matrix3DMeshVersion transpose() const;
  scalar det() const;

  Matrix3DMeshVersion adjoint() const;
  Matrix3DMeshVersion inverse() const;

  /* Accessors */
  Vector3DMeshVersion&       operator[](int i)       { return row[i]; }
  const Vector3DMeshVersion& operator[](int i) const { return row[i]; }

  scalar& operator()(int i, int j)       { return row[i][j]; }
  scalar  operator()(int i, int j) const { return row[i][j]; }

  Vector3DMeshVersion col(int i) const {return Vector3DMeshVersion(row[0][i],row[1][i],row[2][i]);} 
};


/*
 * Non-member prototypes
 */
void transpose(Matrix3DMeshVersion & m);
Matrix3DMeshVersion outer(const Vector3DMeshVersion &u, const Vector3DMeshVersion &v);
Matrix3DMeshVersion outer(const Vector3DMeshVersion &u);
Matrix3DMeshVersion operator*(scalar f, const Matrix3DMeshVersion &m);
Matrix3DMeshVersion Star(const Vector3DMeshVersion &a);



/* 
 * Constructors
 */
inline Matrix3DMeshVersion::Matrix3DMeshVersion(scalar x1y1, scalar x1y2, scalar x1y3,
			  scalar x2y1, scalar x2y2, scalar x2y3,
			  scalar x3y1, scalar x3y2, scalar x3y3){
  row[0][0] = x1y1;
  row[0][1] = x1y2;
  row[0][2] = x1y3;

  row[1][0] = x2y1;
  row[1][1] = x2y2;
  row[1][2] = x2y3;

  row[2][0] = x3y1;
  row[2][1] = x3y2;
  row[2][2] = x3y3;
}


inline Matrix3DMeshVersion::Matrix3DMeshVersion(const Vector3DMeshVersion& r0, const Vector3DMeshVersion& r1, const Vector3DMeshVersion& r2){
  row[0] = r0;
  row[1] = r1;
  row[2] = r2;
}


/*
 * Identity matrices
 */
inline Matrix3DMeshVersion Matrix3DMeshVersion::I(){
  return Matrix3DMeshVersion(1.0, 0.0, 0.0,
		  0.0, 1.0, 0.0,
		  0.0, 0.0, 1.0);
}

inline Matrix3DMeshVersion Matrix3DMeshVersion::Zero(){
  return Matrix3DMeshVersion(0.0, 0.0, 0.0,
		  0.0, 0.0, 0.0,
		  0.0, 0.0, 0.0);
}



/* Matrix Vector Product */
inline Vector3DMeshVersion Matrix3DMeshVersion::operator*(const Vector3DMeshVersion & v) const{
  return Vector3DMeshVersion(row[0] * v, row[1] * v, row[2] * v);
}

/* Matrix Addition */
inline Matrix3DMeshVersion Matrix3DMeshVersion::operator+(const Matrix3DMeshVersion & m) const{
  Matrix3DMeshVersion result(*this);

  result[0] += m[0];
  result[1] += m[1];
  result[2] += m[2];

  return result;
}

inline Matrix3DMeshVersion& Matrix3DMeshVersion::operator+=(const Matrix3DMeshVersion & m){
  row[0] += m[0];
  row[1] += m[1];
  row[2] += m[2];

  return *this;
}


/* Matrix Subtraction */
inline Matrix3DMeshVersion Matrix3DMeshVersion::operator-(const Matrix3DMeshVersion & m) const{
  Matrix3DMeshVersion result(*this);

  result[0] -= m[0];
  result[1] -= m[1];
  result[2] -= m[2];

  return result;
}

inline Matrix3DMeshVersion& Matrix3DMeshVersion::operator-=(const Matrix3DMeshVersion & m){
  row[0] -= m[0];
  row[1] -= m[1];
  row[2] -= m[2];

  return *this;
}


/* Scalar Multiplication */
inline Matrix3DMeshVersion Matrix3DMeshVersion::operator*(const scalar f) const{
  Matrix3DMeshVersion result(*this);

  result[0] *= f;
  result[1] *= f;
  result[2] *= f;

  return result;
}

inline Matrix3DMeshVersion& Matrix3DMeshVersion::operator*=(const scalar f){
  row[0] *= f;
  row[1] *= f;
  row[2] *= f;

  return *this;
}

/* Scalar Division */
inline Matrix3DMeshVersion Matrix3DMeshVersion::operator/(const scalar f) const{
  Matrix3DMeshVersion result(*this);
  scalar f_inv = (scalar) 1.0 / f;

  result[0] *= f_inv;
  result[1] *= f_inv;
  result[2] *= f_inv;

  return result;
}

inline Matrix3DMeshVersion& Matrix3DMeshVersion::operator/=(scalar f){
  Matrix3DMeshVersion result(*this);
  scalar f_inv = (scalar) 1.0 / f;

  row[0] *= f_inv;
  row[1] *= f_inv;
  row[2] *= f_inv;

  return *this;
}


/* Matrix Multiplication */
inline Matrix3DMeshVersion Matrix3DMeshVersion::operator*(const Matrix3DMeshVersion &m) const{
  const Matrix3DMeshVersion mt = m.transpose();
 
  return Matrix3DMeshVersion(row[0] * mt[0], row[0] * mt[1], row[0] * mt[2],
		  row[1] * mt[0], row[1] * mt[1], row[1] * mt[2],
		  row[2] * mt[0], row[2] * mt[1], row[2] * mt[2]);
}


/* Matrix determinate */
inline scalar Matrix3DMeshVersion::det() const{
  return row[0] * cross(row[1],row[2]);
}


/* trace */
inline scalar Matrix3DMeshVersion::trace() const{
  return row[0][0] + row[1][1] + row[2][2];
}


/* Adjoint and Inverse */
inline Matrix3DMeshVersion Matrix3DMeshVersion::adjoint() const{
  return Matrix3DMeshVersion(cross(row[1],row[2]),
		  cross(row[2],row[0]),
		  cross(row[0],row[1]));
}


inline Matrix3DMeshVersion Matrix3DMeshVersion::inverse() const{
  Matrix3DMeshVersion adj(adjoint());

  scalar det = adj[0] * row[0];

  adj = adj.transpose();
  adj /= det;

  return adj;
}

/* transpose */
inline Matrix3DMeshVersion Matrix3DMeshVersion::transpose() const{
  return Matrix3DMeshVersion(row[0][0],row[1][0],row[2][0],
		  row[0][1],row[1][1],row[2][1],
		  row[0][2],row[1][2],row[2][2]);
}



/* Non member functions */
inline void transpose2(Matrix3DMeshVersion & m){
  scalar temp;

  temp = m[0][1];
  m[0][1] = m[1][0];
  m[1][0] = temp;

  temp = m[0][2];
  m[0][2] = m[2][0];
  m[2][0] = temp;

  temp = m[1][2];
  m[1][2] = m[2][1];
  m[2][1] = temp;
}


	
inline Matrix3DMeshVersion outer(const Vector3DMeshVersion &u, const Vector3DMeshVersion &v){
  return Matrix3DMeshVersion(u.X() * v.X(), u.X() * v.Y(), u.X() * v.Z(),
		  u.Y() * v.X(), u.Y() * v.Y(), u.Y() * v.Z(),
		  u.Z() * v.X(), u.Z() * v.Y(), u.Z() * v.Z());
}

inline Matrix3DMeshVersion outer(const Vector3DMeshVersion &u){
  return Matrix3DMeshVersion(u.X() * u.X(), u.X() * u.Y(), u.X() * u.Z(),
		  u.Y() * u.X(), u.Y() * u.Y(), u.Y() * u.Z(),
		  u.Z() * u.X(), u.Z() * u.Y(), u.Z() * u.Z());
}


inline Matrix3DMeshVersion operator*(scalar f, const Matrix3DMeshVersion &m){
  Matrix3DMeshVersion p = m;

  p *= f;

  return p;
}
  
inline Matrix3DMeshVersion Star(const Vector3DMeshVersion &a){
  return Matrix3DMeshVersion( 0,      -a.Z(), a.Y(),
		   a.Z(),  0,     -a.X(),
		   -a.Y(), a.X(),  0);
}


inline std::ostream &operator<<(std::ostream &output, const Matrix3DMeshVersion& m){
  output << m[0] << " ";
  output << m[1] << " ";
  output << m[2];

  return output;
}

inline std::istream &operator>>(std::istream &input, Matrix3DMeshVersion& m){
  input >> m[0];
  input >> m[1];
  input >> m[2];

  return input;
}

#endif
