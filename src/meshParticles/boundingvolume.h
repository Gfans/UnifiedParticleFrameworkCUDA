#ifndef BOUNDINGVOLUME_H
#define BOUNDINGVOLUME_H

#include "vector3d.h"
#include "renderable.h"

class BoundingVolume : Renderable {
 public:
  virtual void Render() {}
};

class AABoundingBox : public BoundingVolume{
 private:
  bool initialized;
 public:

  Vector3DMeshVersion v0;
  Vector3DMeshVersion v1;
  
  AABoundingBox() : initialized(false) {}
  AABoundingBox(const Vector3DMeshVersion& a, const Vector3DMeshVersion& b) : initialized(true), v0(a), v1(b) {}

  void include(const Vector3DMeshVersion& v);
  void include(const AABoundingBox& b);

  virtual void Render();
};


class BoundingSphere : public BoundingVolume {
 private:
  bool initialized;
 public:

  Vector3DMeshVersion center;
  scalar radius;

  BoundingSphere() : initialized(false) {}
  BoundingSphere(const Vector3DMeshVersion& c, const scalar r) : initialized(true), center(c), radius(r) {}

  void include(const Vector3DMeshVersion& v);
  void include(const BoundingSphere& b);
 
  virtual void Render();
};

#endif

