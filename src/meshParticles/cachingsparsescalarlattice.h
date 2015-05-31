#ifndef CACHINGSPARSESCALARLATTICE_H
#define CACHINGSPARSESCALARLATTICE_H


#include "sparsescalarlattice.h"


class CachingSparseScalarLattice : public SparseScalarLattice{
 protected:

  struct V8{ Vector3DMeshVersion vals[8]; };
  struct S8{ scalar   vals[8]; };

#ifdef WIN32
typedef hash_map< Cell, V8, cell_hash_compare> VectorCache;
typedef hash_map< Cell, V8, cell_hash_compare> ScalarCache;
#else
typedef hash_map< Cell, V8, hash_cell, equal_cell > VectorCache;
typedef hash_map< Cell, S8, hash_cell, equal_cell > ScalarCache;
#endif

  VectorCache n_cache;
  ScalarCache s_cache;

 public:
  CachingSparseScalarLattice(const SparseScalarLattice& ssl) : SparseScalarLattice(ssl) {}

  scalar   InterpValue(const  Vector3DMeshVersion& v);
  Vector3DMeshVersion InterpNormal(const Vector3DMeshVersion& v);
  scalar   InterpMeanCurvature(const Vector3DMeshVersion& v);

};

#endif
