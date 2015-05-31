#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#include "vector3d.h"
#include "matrix4d.h"

#include "opengl.h"

const GLuint * nearest_hit(const GLuint buffer[], const GLint hits);
void Matrix4dToOpenGLMatrix(const Matrix4DMeshVersion& a, GLfloat  b[16]);
GLvoid set_material_color ( GLfloat r, GLfloat g, GLfloat b );
scalar depth_z(int wx, int wy);
Vector3DMeshVersion unproject_pixel(int wx, int wy);
Vector3DMeshVersion unproject_pixel(int wx, int wy, GLfloat depth_z);

class GLStatus{
  static void report_stacks();
  static void report_errors(const char *msg);
};


void glMultMatrix(const Matrix4DMeshVersion&);
void glTranslate(const Vector3DMeshVersion&);
void glVertex(const Vector3DMeshVersion&);
void glNormal(const Vector3DMeshVersion&);


#endif
