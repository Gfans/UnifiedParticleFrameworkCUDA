#include "Mesh.h"

//--------------------------------------------------------------------
Mesh::Mesh()
	//--------------------------------------------------------------------
{
	triangle_count = 0;
	vertex_count = 0;
	tex_coord_count = 0;
	normal_count = 0;
}

//--------------------------------------------------------------------
Mesh::~Mesh()
	//--------------------------------------------------------------------
{
	delete[] vertex_positions;
	delete[] vertex_normals;
	delete[] vertex_tex_coords;
	delete[] indices;
	delete[] triangles;
	delete[] vertices;
	delete[] texture_coordinates;
	delete[] normals;
}
