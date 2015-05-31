#ifndef ADF_H
#define ADF_H

#include "hashing.h"
#include "scalarlattice.h"
#include "triangle.h"

#include "renderable.h"

class OctreeNode{
 public:
  bool is_leaf;
  Vector3DMeshVersion corner;
  scalar edge_length;
  union{
    scalar distances[8];
    OctreeNode * children[8];
  };
};

#ifdef WIN32
typedef hash_map<Cell, OctreeNode * ,cell_hash_compare> _ADF;
#else
typedef hash_map<Cell, OctreeNode * , hash_cell, equal_cell > _ADF;
#endif 


class ADF : public _ADF, public Renderable {
 protected:
  scalar voxelsize;
  scalar inv_voxelsize;

 public:
  ADF(const scalar& vsize) { assert(vsize > 0); voxelsize = vsize; inv_voxelsize = 1.0 / vsize; }
  virtual ~ADF(){}

  scalar VoxelSize() const { return voxelsize; }

  static const Cell TraversalOrder[8];

  virtual void Render();
};

#endif
