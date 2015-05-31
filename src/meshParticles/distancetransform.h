#ifndef DISTANCETRANSFORM_H
#define DISTANCETRANSFORM_H

#include "trianglemesh.h"
#include "trianglenearestpoint.h"
#include "sparsescalarlattice.h"
#include "adf.h"



class TriangleSignedDistance : private TriangleNearestPoint {
 public:
  Vector3DMeshVersion angle_weighted_normals[7];

  TriangleSignedDistance(const Triangle& tri) : TriangleNearestPoint(tri) {}

  scalar SignedDistanceTo(const Vector3DMeshVersion& p) const;
};

class TriangleSquaredDistance : private TriangleNearestPoint {
 public:
  TriangleSquaredDistance(const Triangle& tri) : TriangleNearestPoint(tri) {}

  scalar SquaredDistanceTo(const Vector3DMeshVersion& p) const;
};


class SignedDistanceTransform {
 private:
  std::vector< TriangleSignedDistance > tri_distances;
  //  std::vector< AABoundingBox > bounding_boxes;
  std::vector< Triangle > triangles;
    
 public:
  SignedDistanceTransform(const TriangleMesh& mesh);

  void Voxelize(SparseScalarLattice& lattice, scalar max_dist);
  void Voxelize(ADF& adf, SparseScalarLattice& lattice, scalar max_dist, int max_subdivisions, scalar abs_error);
  void Voxelize(ADF& adf, scalar max_dist, int max_subdivisions, scalar abs_error);
 
  static void Test();
};


class UnsignedDistanceTransform { 
  private:
  std::vector< TriangleSquaredDistance > tri_distances;
  std::vector< Triangle > triangles;
 public:
  UnsignedDistanceTransform(const TriangleMesh& mesh);
  void Voxelize(SparseScalarLattice& lattice, scalar max_dist);
};

#endif
