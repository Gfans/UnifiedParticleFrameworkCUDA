#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "boundingvolume.h"
#include "vector3d.h"
#include "ray.h"
#include "renderable.h"


class TriangleException {};
class DegenerateTriangleException : public TriangleException {};

class Triangle : public Renderable {
  friend std::ostream &operator<<(std::ostream&, const Triangle&);
  friend std::istream &operator>>(std::istream &input, Triangle& v);
 protected:
  Vector3DMeshVersion corners[3];
 public:
  Triangle(){}
  Triangle(const Vector3DMeshVersion& v0, const Vector3DMeshVersion& v1, const Vector3DMeshVersion& v2);
  Triangle(const Vector3DMeshVersion& v0, const Vector3DMeshVersion& v1, const Vector3DMeshVersion& v2, const Vector3DMeshVersion& n0, const Vector3DMeshVersion& n1, const Vector3DMeshVersion& n2);

  Vector3DMeshVersion Bary(scalar a, scalar b, scalar c) const;
  scalar Perimeter() const;
  scalar SurfaceArea() const;
  Vector3DMeshVersion RandomSample() const;

  const Vector3DMeshVersion& A() const {return corners[0]; }
        Vector3DMeshVersion& A()       {return corners[0]; }
  const Vector3DMeshVersion& B() const {return corners[1]; }
        Vector3DMeshVersion& B()       {return corners[1]; }
  const Vector3DMeshVersion& C() const {return corners[2]; }
        Vector3DMeshVersion& C()       {return corners[2]; }

  const Vector3DMeshVersion& operator[](int i) const{ return corners[i]; }
        Vector3DMeshVersion& operator[](int i)      { return corners[i]; }


	
  // The non-unitized normal of this triangle
  Vector3DMeshVersion Normal() const;

  bool Intersects(const Ray& ray) const;
  bool Intersection(const Ray& ray, scalar& t) const;

  AABoundingBox BoundingBox() const;

  void Translate(const Vector3DMeshVersion& v);

  virtual void Render();
};

inline std::ostream &operator<<(std::ostream &output, const Triangle& t){
  output << t.corners[0] << " " << t.corners[1] << " " << t.corners[2] << " ";
  //  output << t.normals[0] << " " << t.normals[1] << " " << t.normals[2];

  return output;
}

inline std::istream &operator>>(std::istream &input, Triangle& t){
  input >> t.corners[0];
  input >> t.corners[1];
  input >> t.corners[2];
  return input;
}

#endif

