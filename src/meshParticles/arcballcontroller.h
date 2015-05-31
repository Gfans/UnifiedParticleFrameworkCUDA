#ifndef ARCBALLCONTROLLER_H
#define ARCBALLCONTROLLER_H

#include "baseballcontroller.h"
#include "vector2d.h"

class ArcballController : public BaseballController
{
private:
  Vector2DMeshVersion ball_ctr;
  scalar ball_radius;

  Quaternion q_now, q_down, q_drag;	// Quaternions describing rotation
  Vector3DMeshVersion v_from, v_to;		//
  
  bool is_dragging;

protected:
    Vector3DMeshVersion proj_to_sphere(const Vector2DMeshVersion&);
    void update();

public:
    ArcballController();

    virtual void update_animation();
    virtual bool mouse_down(int *where, int which);
    virtual bool mouse_up(int *where, int which);
    virtual bool mouse_drag(int *where, int *last, int which);

    virtual void apply_transform();
    virtual void get_transform(Vector3DMeshVersion & c, Vector3DMeshVersion &t, Quaternion & q);
    virtual void set_transform(const Vector3DMeshVersion & c, const Vector3DMeshVersion & t, const Quaternion & q); 

    virtual void write(std::ostream&);
    virtual void read(std::istream&);
};



#endif
