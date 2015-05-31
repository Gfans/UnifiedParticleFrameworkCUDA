#ifndef BODY_H
#define BODY_H

#include <vector>
#include <sstream>

#include "libbell.h"
#include "vector3d.h"
#include "material.h"

#include "renderable.h"

#include "particle.h"
#include "particlehashmap.h"

class Body : public Renderable{
 protected:
  bool is_dynamic;
 public:
  
  Material material;

  virtual ~Body() {}

/*   Material& Mat()                  { return material; } */
/*   void      Mat(const Material& m) { material = m; } */

  virtual void            SetMaterial(const Material& m) { material = m; }
  virtual const Material& GetMaterial() const { return material; }

  virtual scalar RequiredCellSize() const = 0;

  virtual void addToMap(ParticleHashMap * emap) = 0;
  virtual void getParticles(vector< Particle * >& pvec)  {}

  bool IsDynamic() const{ return is_dynamic; }
  virtual void SetDynamic(bool new_value) { is_dynamic = new_value; }

  virtual void StateToStream(ostream& output) const{
    output << is_dynamic << " ";
    material.StateToStream(output);
  }
  virtual void StreamToState(istream& input){
    input >> is_dynamic;
    material.StreamToState(input);
  }

  virtual uint flattenedSize() const = 0;

  virtual scalar MaxTimestep() const { return SCALAR_MAX; }

  virtual void Update(scalar time) {}

  virtual void EvaluateForces(scalar time, scalar timestep) = 0;

  virtual void StateToArray(scalar * base) = 0;
  virtual void ArrayToState(scalar * base) = 0;
  virtual void DerivsToArray(scalar * base) = 0;


  virtual Particle& operator[](unsigned int i) = 0;
};

#endif
