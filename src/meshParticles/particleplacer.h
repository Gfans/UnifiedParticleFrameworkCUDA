#ifndef PARTICLEPLACER_H
#define PARTICLEPLACER_H

#include "particle.h"
#include "trianglemesh.h"

class ParticlePlacer {
 public:
  static void CoverMeshWithParticles(const TriangleMesh& mesh, std::vector< Particle >& particles, scalar radius, scalar density);
  static void CoverMeshSurfaceWithParticles(const TriangleMesh& mesh, std::vector< Particle >& particles, scalar radius, scalar density);

  static void FillMeshWithParticles(const TriangleMesh& mesh, std::vector< Particle >& particles, scalar radius);
};

#endif
