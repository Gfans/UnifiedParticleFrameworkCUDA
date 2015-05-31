#ifndef BASEBALL_H
#define BASEBALL_H

#include <iostream>

#include "quaternion.h"
#include "vector3d.h"
#include "opengl_utils.h"

class BaseballController
{
public:
  Vector3DMeshVersion ctr;			// Describes bounding sphere of object
  scalar radius;		//

  Quaternion curquat;		// Current rotation of object
  Vector3DMeshVersion trans;			// Current translation of object

public:
    BaseballController();

    // Required initialization method
    void bounding_sphere(const Vector3DMeshVersion& v, scalar r) {ctr = v; radius = r;}

    // Standard event interface provide by all Ball controllers
    virtual void update_animation() = 0;
    virtual bool mouse_down(int *where, int which) = 0;
    virtual bool mouse_up(int *where, int which) = 0;
    virtual bool mouse_drag(int *where, int *last, int which) = 0;

    // Interface for use during drawing to apply appropriate transformation
    virtual void apply_transform();
    virtual void unapply_transform();

    // Interface for reading/writing transform
    virtual void write(std::ostream&);
    virtual void read(std::istream&);
};


inline BaseballController::BaseballController()
{
    curquat = Quaternion::ident();

    trans = Vector3DMeshVersion(0,0,0);
    ctr = Vector3DMeshVersion(0,0,0);
    radius=1;
}

inline void BaseballController::apply_transform()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glTranslate(trans);

    glTranslate(ctr);

    const Matrix4DMeshVersion M=curquat.unitToMatrix4d();
    
    glMultMatrix(M);

    glTranslate(-ctr);
}

inline void BaseballController::unapply_transform()
{
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

inline void BaseballController::write(std::ostream& out)
{
    out << "baseball ";
    out << curquat << " " << trans << " " << ctr << " " << radius << std::endl;
}

inline void BaseballController::read(std::istream& in)
{
    std::string name;

    in >> name;
    in >> curquat >> trans >> ctr >> radius;
}



#endif
