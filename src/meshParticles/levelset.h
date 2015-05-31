#ifndef LEVELSET_H
#define LEVELSET_H

#include <queue>

#include "cell.h"
#include "triangle.h"
#include "hashing.h"
#include "sparsescalarlattice.h"
#include "renderable.h"


class ArrivalPair{
 public:
  scalar time;
  Cell   location;
  ArrivalPair(scalar t, const Cell& c){
    time     = t;
    location = c;
  }
};

struct CompareArrivalTimes{
  bool operator()(const ArrivalPair& a, const ArrivalPair& b){
    return fabsf(a.time) > fabsf(b.time);
  }
};

typedef std::priority_queue< ArrivalPair, std::vector<ArrivalPair>, CompareArrivalTimes > FMMQueue;


class LevelSet : public Renderable {
 protected:
  scalar cellsize;
  scalar cellsize_squared;
  scalar inv_cellsize;
  scalar inv_cellsize_squared;
  scalar time;

  SparseScalarLattice lattice;

  CellSet frozen_set;
  CellSet narrowband_set;

 public:
  LevelSet(scalar cell_size);
  LevelSet(const SparseScalarLattice& lat);
  virtual ~LevelSet();
  
  void InnerVolume(CellSet& cells);

  void DistanceInnerVolume();
  void Redistance(unsigned int num_voxels = 4);
  void RedistanceTo(scalar distance);

  scalar Recompute(const Cell& pi) const;

  scalar   VoxelWidth() const { return cellsize; }

  const CellSet&        getNarrowBandSet() const { return narrowband_set; }


  //  void IterateUntilConvergence(scak

  void AdvanceForwardEuler(scalar delta_t = 0.0);
  scalar ForwardEulerSingleStep();
  void Triangulate(scalar isosurface, std::vector< Triangle >& triangles, bool smooth = false) const;
  virtual void Render();

  SparseScalarLattice& GetLattice() {return lattice;}

  
 protected:
  bool IsNearZeroIsosurface(const Cell& center) const;
  scalar SignedDistanceToInterface(const Cell& pi) const;

  virtual scalar DerivAt(const Cell& pi, scalar time, scalar& allowable_timestep){ return 0;}

  bool isFrozen(const Cell& pi) const;
  bool hasValue(const Cell& pi) const;

  std::string StateToString() const;
  void StringToState(std::string s);
};


inline int solve_quadratic(scalar coeff[3], scalar sol[2]){
  if(coeff[2] == 0.0){
    if(coeff[1] == 0.0){
      return 0;    
    }
    else {
      sol[0] = -coeff[0] / coeff[1];
      return 1;
    }
  }

  scalar a = coeff[2];
  scalar b = coeff[1];
  scalar c = coeff[0];

  scalar discriminant = b*b - 4*a*c;

  if(discriminant < 0.0){
    return 0;
  }
  else if(discriminant < SCALAR_EPSILON){
    sol[0] = -b/(2*a);
    return 1;
  }
  else {
    scalar D = sqrtf(discriminant);
    sol[0] = (-b - D)/(2*a);
    sol[1] = (-b + D)/(2*a);
    return 2;
  }
}

#endif
