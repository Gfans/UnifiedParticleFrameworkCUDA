#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "opengl_utils.h"

class Renderable {
 public:
  virtual void Render() = 0;
};


#endif
