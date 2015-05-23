#ifndef GPU_UNIFIED_MESH_H_
#define GPU_UNIFIED_MESH_H_ 

#include "vmmlib/vector3.h"

/**
* A mesh used by the ObjLoader class. This is a reused class; most of its
* attributes are not used.
*/
struct Mesh
{
	Mesh();

	~Mesh();

	struct Vertex
	{
		Vertex(){}
		Vertex(float x, float y, float z) : coordinates(x, y, z){}
		Vertex(const Vertex& rhs) : coordinates(rhs.coordinates){}	
		Vertex& operator=(const Vertex& rhs) {coordinates = rhs.coordinates; return *this;}

		vmml::Vector3f coordinates;
	};

	struct Triangle
	{
		int tex_coord_indices[3];
		int vertex_indices[3];
		int normal_indices[3];
	};

	struct TextureCoordinate
	{
		float s;
		float t;
	};

	Vertex* vertex_positions;
	vmml::Vector3f* vertex_normals;
	TextureCoordinate* vertex_tex_coords;

	unsigned int* indices;

	unsigned int triangle_count;
	Triangle* triangles;

	unsigned int vertex_count;
	Vertex* vertices;

	unsigned int tex_coord_count;
	TextureCoordinate* texture_coordinates;

	unsigned int normal_count;
	vmml::Vector3f* normals;
};

#endif	// GPU_UNIFIED_MESH_H_
