#ifndef PARTICLE_H
#define PARTICLE_H


#include <vector>
 
#include "vector3d.h"
#include "renderable.h"

class Body;

class Particle{
  friend std::ostream &operator<<(std::ostream &output, const Particle&);
  friend std::istream &operator>>(std::istream &input,        Particle& p);
 protected:
  Vector3DMeshVersion location;
  scalar   radius;

 public: 
  Particle() { radius = 0; }
  Particle(const Vector3DMeshVersion& loc){ location = loc; }
  Particle(const Vector3DMeshVersion& loc, const scalar rad){ location = loc; radius = rad; }

  const Vector3DMeshVersion&  Location() const{ return location; }
  void  SetLocation(const Vector3DMeshVersion& new_location) { location = new_location; }

  const scalar&  Radius() const{ return radius; }
  void  SetRadius(const scalar& new_radius) { radius = new_radius; }

  virtual void Render();
};


inline std::ostream &operator<<(std::ostream &output, const Particle& p){
  output << p.location << " " << p.radius;
  return output;
}

inline std::istream &operator>>(std::istream &input, Particle& p){
  input >> p.location >> p.radius;
  return input;
}


inline void Particle::Render(){
  glPushMatrix();
  {
    glTranslate(Location());
    glutSolidSphere(radius,6,4);
  }
  glPopMatrix();
}



#endif
