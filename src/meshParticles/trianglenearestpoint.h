#ifndef TRIANGLENEARESTPOINT_H
#define TRIANGLENEARESTPOINT_H


#include "triangle.h"

class TriangleNearestPoint{
 private:
  Vector3DMeshVersion corners[3];

  Vector3DMeshVersion unitnormal;

  Vector3DMeshVersion span01;
  Vector3DMeshVersion span02;
  Vector3DMeshVersion span12;
  scalar u11, u22, u12, u33, u13;

  scalar d_inv;

  Vector3DMeshVersion u_span01, u_span02, u_span12;

 public:
  TriangleNearestPoint(const Triangle&);

  Vector3DMeshVersion NearestPointTo(const Vector3DMeshVersion& p) const;
  Vector3DMeshVersion NearestPointToVerbose(const Vector3DMeshVersion& p, int& region) const;

  enum VoronoiRegion { CORNER_A_REGION = 0, CORNER_B_REGION = 1, EDGE_AB_REGION = 2, 
		       CORNER_C_REGION = 3, EDGE_AC_REGION = 4, EDGE_BC_REGION = 5, 
		       FACE_REGION = 6, INVALID_REGION = 7 };


  static bool Test();
};


#endif
