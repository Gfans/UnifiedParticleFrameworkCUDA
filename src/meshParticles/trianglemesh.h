#ifndef TRIANGLEMESH2_H
#define TRIANGLEMESH2_H

#include <vector>

#include "renderable.h"

#include "boundingvolume.h"
#include "triangle.h"
#include "matrix3d.h"
#include "sparsescalarlattice.h"
#include "index.h"





class TriangleMesh : public Renderable {
  friend std::ostream &operator<<(std::ostream &output, const TriangleMesh&);
  friend std::istream &operator>>(std::istream &input,        TriangleMesh&);
 public:
  std::vector< Vector3DMeshVersion > vertices;
  std::vector< IndexTriple > faces;

  std::vector< Vector3DMeshVersion > normals;
  std::vector< IndexTriple > face_normals;

 public:




  class Edge{
  public:
    Index I, J;
    Edge(const Index& first, const Index& second) : I(first), J(second) {}

    bool operator==(const Edge &e) const { return (I == e.I) && (J == e.J);}
    bool operator!=(const Edge &e) const { return (I != e.I) || (J != e.J);}
  };


#ifdef WIN32

  struct edge_hash_compare{
    enum
      {
	bucket_size = 4,
	min_buckets = 8
      };
    size_t operator()(const Edge& e) const{
      return 2547483661 * e.I + 3047483653 * e.J;
    }
    bool operator()(const Edge& a, const Edge& b) const{
      if(a.I < b.I)
	return true;
      if(a.I > b.I)
	return false;
      if(a.J < b.J)
	return true;
      return false;    
    }
  };
  
  typedef hash_set<Edge, edge_hash_compare> EdgeSet;
  
  
#else
  struct hash_edge{
    size_t operator()(const TriangleMesh::Edge& e) const{
      return 2547483661 * e.I + 3047483653 * e.J;
    }
  };
  
  struct equal_edge{
    bool operator()(const TriangleMesh::Edge& a, const TriangleMesh::Edge& b) const{
      return a == b;
    }
  };
  typedef hash_set< TriangleMesh::Edge, hash_edge, equal_edge > EdgeSet;
#endif
  
  



  TriangleMesh(){}
  TriangleMesh(const std::string filename);
  virtual ~TriangleMesh();

  Triangle operator[](unsigned int i) const;
  Index size() const { return faces.size(); }

  scalar Volume() const;
  void MassProperties (scalar& volume, Vector3DMeshVersion& cm, Matrix3DMeshVersion& inertia) const;
  void Edges(std::vector<Edge>& edges);

  static TriangleMesh UnitCube();

  void Translate(const Vector3DMeshVersion& v);
  void Scale(const Vector3DMeshVersion& v);
  void Transform(const Matrix3DMeshVersion& m);

  AABoundingBox GetAABoundingBox() const;
  BoundingSphere GetBoundingSphere() const;

/*   void Intersect(const Ray& r, std::vector< scalar >& intersections) const; */

  void LoadFromOBJFile(const std::string& filename);
  void LoadFromSMESHFile(const std::string& filename);

  void SaveToOBJFile(const std::string& filename);

  virtual void Render();
};



void LoadMultipleFromOBJFile(const std::string& filename, std::vector<TriangleMesh>& meshes, std::vector<std::string>& groupnames);

inline std::ostream &operator<<(std::ostream &output, const TriangleMesh& m){
  output << m.vertices.size() << " ";

  for(unsigned int i = 0; i < m.vertices.size(); i++){
    output << m.vertices[i] << " ";
  }

  output << m.faces.size() << " ";

  for(unsigned int i = 0; i < m.faces.size(); i++){ 
    output << m.faces[i] << " "; 
  } 

  output << m.normals.size() << " ";

  for(unsigned int i = 0; i < m.normals.size(); i++){ 
    output << m.normals[i] << " "; 
  } 

  output << m.face_normals.size() << " ";

  for(unsigned int i = 0; i < m.face_normals.size(); i++){ 
    output << m.face_normals[i] << " "; 
  } 

  return output;
}

inline std::istream &operator>>(std::istream &input, TriangleMesh& m){

  unsigned int num_vertices;  
  input >> num_vertices;  
  m.vertices.resize(num_vertices);  
  for(unsigned int i = 0; i < num_vertices; i++){
    input >> m.vertices[i];
  }

  unsigned int num_faces;
  input >> num_faces;
  m.faces.resize(num_faces);
  for(unsigned int i = 0; i < num_faces; i++){ 
    input >> m.faces[i]; 
  }   

  unsigned int num_normals;
  input >> num_normals;
  m.normals.resize(num_normals);
  for(unsigned int i = 0; i < num_normals; i++){ 
    input >> m.normals[i]; 
  }   

  unsigned int num_face_normals;
  input >> num_face_normals;
  m.face_normals.resize(num_face_normals);
  for(unsigned int i = 0; i < num_face_normals; i++){ 
    input >> m.face_normals[i]; 
  }   

  return input;
}


#endif
