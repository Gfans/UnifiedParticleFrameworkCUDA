#ifndef TRIANGLEBOXOVERLAP_H
#define TRIANGLEBOXOVERLAP_H

#include "triangle.h"

class TriangleBoxOverlap{
 private:
  Vector3DMeshVersion corners[3];

  Vector3DMeshVersion edge01;
  Vector3DMeshVersion edge12;
  Vector3DMeshVersion edge20;

  Vector3DMeshVersion bbox_min;
  Vector3DMeshVersion bbox_max;

  Vector3DMeshVersion normal;

///  AABoundingBox bbox;
 public:
  TriangleBoxOverlap(const Triangle&);

  bool Overlaps(const Vector3DMeshVersion& box_center, const Vector3DMeshVersion& boxhalfsize);

  static bool Test();
};


#endif
