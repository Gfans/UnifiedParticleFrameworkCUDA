#ifndef CELL_H
#define CELL_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "libbell.h"
#include "vector3d.h"
#include "hashing.h"


class Cell{
  friend std::ostream &operator<<(std::ostream&, const Cell&);
  friend std::istream &operator>>(std::istream &input, Cell& c);
 private:
  int location[3];
 public:
  Cell(int x = 0, int y = 0, int z = 0){
    location[0] = x;
    location[1] = y;
    location[2] = z;
  }

  Cell(const int x[3]){
    location[0] = x[0];
    location[1] = x[1];
    location[2] = x[2];
  }

  /* Access Specific entries */
  int  operator[](int i) const{ return location[i]; }
  int& operator[](int i){ return location[i]; }

  int X() const{ return location[0]; }
  int Y() const{ return location[1]; }
  int Z() const{ return location[2]; }

  Vector3DMeshVersion toVector3d() const{ 
    return Vector3DMeshVersion((scalar) X(),(scalar) Y(), (scalar) Z());
  }

  size_t hash() const { return  150001*X() + 157057*Y() + 170003*Z(); }

};  


inline Cell operator+(const Cell &a, const Cell& b) { return Cell(a.X() + b.X(), a.Y() + b.Y(), a.Z() + b.Z()); }
inline Cell operator-(const Cell &a, const Cell& b) { return Cell(a.X() - b.X(), a.Y() - b.Y(), a.Z() - b.Z()); }
inline bool operator==(const Cell &a, const Cell& b) { return (a.X() == b.X()) && (a.Y() == b.Y()) && (a.Z() == b.Z());}
inline bool operator!=(const Cell &a, const Cell& b) { return (a.X() != b.X()) || (a.Y() != b.Y()) || (a.Z() != b.Z());}

inline bool operator<(const Cell& a, const Cell& b){
  if (a.X() < b.X()) return true;
  if (a.X() > b.X()) return false;
  
  if (a.Y() < b.Y()) return true;
  if (a.Y() > b.Y()) return false;
  
  return a.Z() < b.Z();
}




inline Cell toCell(const Vector3DMeshVersion& v, scalar cell_size){
  scalar cell_size_inverse = (scalar)1.0 / cell_size;
  return Cell( FLOOR((scalar) v.X() * cell_size_inverse),
	       FLOOR((scalar) v.Y() * cell_size_inverse),
	       FLOOR((scalar) v.Z() * cell_size_inverse));

}

inline Cell toCellInverse(const Vector3DMeshVersion& v, scalar cell_size_inverse){
  return Cell( FLOOR((scalar) v.X() * cell_size_inverse),
	       FLOOR((scalar) v.Y() * cell_size_inverse),
	       FLOOR((scalar) v.Z() * cell_size_inverse));
}


inline std::ostream &operator<<(std::ostream &output, const Cell& cell){
  return output << cell.X() << " " << cell.Y() << " " << cell.Z();
}

inline std::istream &operator>>(std::istream &input, Cell& c){
  input >> c[0];
  input >> c[1];
  input >> c[2];
  
  return input;
}




#ifdef WIN32
struct cell_hash_compare{
   enum{
      bucket_size = 4,
      min_buckets = 8
   };

   size_t operator()(const Cell& c) const{
     return  c.hash();
   }

   bool operator()(const Cell& c1, const Cell& c2) const{
     return c1 < c2;
   }
};


typedef hash_set<Cell, cell_hash_compare> CellSet;

#else

struct hash_cell{
size_t operator()(const Cell& c) const{
return c.hash();
}
};

struct equal_cell{
  bool operator()(const Cell& a, const Cell& b) const{
    return a == b;
  }
};

typedef hash_set< Cell, hash_cell, equal_cell > CellSet;

#endif





#endif
