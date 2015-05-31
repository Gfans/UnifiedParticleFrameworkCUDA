#ifndef EXTRACTSURFACE_H
#define EXTRACTSURFACE_H

#include "sparsescalarlattice.h"
#include "trianglemesh.h"

//#include "adf.h"
//void ExtractIsosurface(const SparseScalarLattice& lattice, scalar isosurface, std::vector< Triangle >& triangles);
//void ExtractIsosurface(const ADF& adf, scalar isosurface, std::vector< Triangle >& triangles);


void ExtractIsosurface(const SparseScalarLattice& lattice, scalar isosurface, TriangleMesh& mesh);


#endif
