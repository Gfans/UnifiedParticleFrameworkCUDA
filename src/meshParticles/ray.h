#ifndef RAY_H
#define RAY_H

#include "vector3d.h"

class Ray{
 public:
  Vector3DMeshVersion o;
  Vector3DMeshVersion d;

  Ray(){
    o = Vector3DMeshVersion(0,0,0);
    d = Vector3DMeshVersion(0,0,1);
  }

  Ray(Vector3DMeshVersion origin, Vector3DMeshVersion direction){
    o = origin;
    d = direction;
  }


  Vector3DMeshVersion EvalAt(scalar x) const{
    return o + x * d;
  }

  void Render() const;

};

#endif
