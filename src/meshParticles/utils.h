#ifndef UTILS_H
#define UTILS_H

#ifndef M_PI     //VCC doesn't have an M_PI?
#define M_PI 3.14159265358979323846
#endif

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include "libbell.h"
#include "vector3d.h"

inline scalar SphereMassFromDensity(const scalar radius, const scalar density){
  return (scalar)((4.0 / 3.0) * M_PI * pow(radius,(scalar) 3.0) * density);
}

inline void eat_one_char(std::ifstream &file){
  file.get();
}

inline void move_to_next_line(std::ifstream &file){
  char c;
  while(true){
    c = file.get();
    if(file.eof() || c == '\n'){
      return;
    }
  }
}

scalar TetrahedronRadiusRatio(const Vector3DMeshVersion& v0, const Vector3DMeshVersion& v1, const Vector3DMeshVersion& v2, const Vector3DMeshVersion& v3);

std::string GetExtension(const std::string& str);
std::string GetNonExtension(const std::string& str);

bool TestUtils();

#endif
