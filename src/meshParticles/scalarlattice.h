#ifndef SCALARLATTICE_H
#define SCALARLATTICE_H

#include "boundingvolume.h"
#include "vector3d.h"
#include "cell.h"
#include "scalar.h"

class LatticeException{};

class InvalidCellException : public LatticeException{
 private:
  Cell location;
 public:
  InvalidCellException(const Cell& c){
    location = c;
  }
  const Cell& Location() const{ return location; }
};


class ScalarLattice{
 private:
  scalar voxelsize;
  scalar inv_voxelsize;

 public:
  ScalarLattice(scalar vsize){ assert(vsize > 0.0); voxelsize = vsize; inv_voxelsize = (scalar) 1.0/voxelsize; }
  virtual ~ScalarLattice() {;}

  scalar VoxelSize() const { return voxelsize; }
  scalar InverseVoxelSize() const { return inv_voxelsize; }

  virtual bool HasValue(const Cell& c) const = 0;
  virtual bool ContainsPoint(const Vector3DMeshVersion& v) const = 0;
  virtual AABoundingBox GetAABBox() const = 0;

  Cell ToCell(const Vector3DMeshVersion& v) const;

  virtual scalar Value(const Cell& c) const = 0;

  virtual Vector3DMeshVersion Normal(const Cell& c) const;
  virtual scalar   Upwind(const Cell & c, const Vector3DMeshVersion& direction) const;
  virtual Vector3DMeshVersion UpwindNormal(const Cell & c, const Vector3DMeshVersion& direction) const;
  virtual scalar MeanCurvature(const Cell& c) const;

  // Use trilinear interpolation to find desired values
  virtual scalar   InterpValue(const  Vector3DMeshVersion& v) const;
  virtual Vector3DMeshVersion InterpNormal(const Vector3DMeshVersion& v) const;
  virtual scalar   InterpMeanCurvature(const Vector3DMeshVersion& v) const;

  static const Cell N6i[6];
};





#endif
