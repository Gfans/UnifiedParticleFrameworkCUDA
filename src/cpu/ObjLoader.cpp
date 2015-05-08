#include "ObjLoader.h"
#include "Polygon.h"

//--------------------------------------------------------------------
ObjLoader::ObjLoader()
	//--------------------------------------------------------------------
{
}

//--------------------------------------------------------------------
ObjLoader& ObjLoader::GetInstance()
{
	static ObjLoader instance;
	return instance;
}

//--------------------------------------------------------------------
std::vector<Face*> ObjLoader::LoadFaces(const std::string& file_name)
{
	// turn the file into a mesh
	Mesh* mesh;
	try
	{
		mesh = Load(file_name);
	}
	catch(const FileNotFound& fnf)
	{
		throw fnf;
	}
	catch(const FileError& fe)
	{
		throw fe;
	}

	// turn the triangles of the mesh into the polygons expected by the caller
	std::vector<Face*> faces;
	if (mesh)
	{
		faces.reserve(mesh->triangle_count);
		for(unsigned int i=0; i<mesh->triangle_count; ++i)
		{
			Polygon* current_polygon = new Polygon();
			Mesh::Triangle* current_triangle = &mesh->triangles[i];
			if (current_polygon && current_triangle)
			{
				for(int j=0; j<3; ++j)
				{
					vmml::Vector3f vertex = mesh->vertices[current_triangle->vertex_indices[j]].coordinates;
					current_polygon->AddVertex(vertex);
				}
				faces.push_back(current_polygon);
			}
		}
	}

	delete mesh;
	return faces;
}

//--------------------------------------------------------------------
Mesh* ObjLoader::Load(const std::string& file_name) const
{
	// The original objloader Load method with some minor changes

	FILE* fp=0;

	fp = fopen(file_name.c_str(),"r");

	if(NULL == fp) 
		throw FileNotFound(file_name);

	/*
	this buffer boosts speed but lines are assumed to be no longer than
	kBufferSize characters.
	*/
	const int kBufferSize = 256;
	char buffer[kBufferSize];

	Mesh* mesh = new Mesh();

	// Verify that allocations succeeded
	if (NULL == mesh)
	{
		fclose(fp);
		fprintf(stderr, "Failed to allocate mesh object!\n");
		return NULL;
	}

	/*
	go through the entire file and count how many times each tag exists
	*/
	while(!feof(fp))
	{
		fgets(buffer,kBufferSize,fp);

		// texture coordinate tag
		if(strncmp("vt ",buffer,3)==0) ++mesh->tex_coord_count;
		else

			// normal tag
			if(strncmp("vn ",buffer,3) == 0 ) ++mesh->normal_count;
			else

				// vertex tag
				if(strncmp("v ",buffer,2)==0) ++mesh->vertex_count;
				else

					// face tag (triangle assumed)
					if(strncmp("f ",buffer,2)==0) mesh->triangle_count++;
	}

	fclose(fp);

	// if the file is degenerated, no model shall be returned at all
	if(mesh->triangle_count == 0 || mesh->vertex_count == 0)
	{
		delete mesh;
		throw FileError(file_name);
	}

	/*
	now that the count of every tag is known, allocate memory for
	all information dynamically
	*/

	// normalized
	//model->meshes=new Mesh[model->meshCount]; // adapted - the model variable is not available here
	mesh->triangles = new Mesh::Triangle[mesh->triangle_count];
	mesh->vertices = new Mesh::Vertex[mesh->vertex_count];
	mesh->normals = new vmml::Vector3f[mesh->normal_count];
	mesh->texture_coordinates = new Mesh::TextureCoordinate[mesh->tex_coord_count];

	// denormalized - for vertex array rendering
	mesh->indices = new unsigned int[mesh->triangle_count*3];
	mesh->vertex_positions = new Mesh::Vertex[mesh->triangle_count*3];
	mesh->vertex_normals = new vmml::Vector3f[mesh->triangle_count*3];
	mesh->vertex_tex_coords = new Mesh::TextureCoordinate[mesh->triangle_count*3];

	// Verify that allocations succeeded
	if (NULL == mesh->triangles || NULL == mesh->vertices || NULL == mesh->normals || NULL == mesh->texture_coordinates 
		|| NULL == mesh->indices || NULL == mesh->vertex_positions || NULL == mesh->vertex_normals || NULL == mesh->vertex_tex_coords)
	{
		fclose(fp);
		fprintf(stderr, "Failed to allocate memories for pointers of mesh!\n");
		return NULL;
	}

	/*
	go through the file a second time and read in all information needed
	into the mesh struct
	*/
	fp = fopen(file_name.c_str(),"r");

	unsigned int vertices_loaded=0;
	unsigned int normals_loaded=0;
	unsigned int texture_coordinates_loaded=0;
	unsigned int triangles_loaded=0;
	//unsigned int textures_loaded=0; // adapted - this is not used here anymore

	while(!feof(fp))
	{
		fgets(buffer,kBufferSize,fp);

		if( strncmp("vn ",buffer,3) == 0 )
		{
			sscanf((buffer+3),"%f%f%f",	&mesh->normals[normals_loaded][0],
				&mesh->normals[normals_loaded][1],
				&mesh->normals[normals_loaded][2]);
			++normals_loaded;
		}
		else

			if(strncmp("vt ",buffer,3) == 0)
			{
				sscanf((buffer+3),"%f%f", 	&mesh->texture_coordinates[texture_coordinates_loaded].s,
					&mesh->texture_coordinates[texture_coordinates_loaded].t);
				++texture_coordinates_loaded;
			}
			else

				if( strncmp("v ",buffer,2) == 0 )
				{
					sscanf((buffer+2),"%f%f%f", &mesh->vertices[vertices_loaded].coordinates[0],
						&mesh->vertices[vertices_loaded].coordinates[1],
						&mesh->vertices[vertices_loaded].coordinates[2]);
					++vertices_loaded;
				}
				else

					if( strncmp("f ",buffer,2) == 0 )
					{
						Mesh::Triangle* triangle = &mesh->triangles[triangles_loaded];

						//read all data for this triangle	(normalized)				
						sscanf(buffer+2, "%d/%d/%d %d/%d/%d %d/%d/%d",	&triangle->vertex_indices[0],
							&triangle->tex_coord_indices[0],
							&triangle->normal_indices[0],
							&triangle->vertex_indices[1],
							&triangle->tex_coord_indices[1],
							&triangle->normal_indices[1],
							&triangle->vertex_indices[2],
							&triangle->tex_coord_indices[2],
							&triangle->normal_indices[2]);

						//reduce the indices by 1 because array indices start at 0, while obj indices starts at 1
						for(int i=0;i<3;++i)
						{
							--triangle->tex_coord_indices[i];
							--triangle->normal_indices[i];
							--triangle->vertex_indices[i];
						}

						//read all data for this triangle	(denormalized for vertex array rendering)
						mesh->vertex_positions[triangles_loaded*3+0]=mesh->vertices[triangle->vertex_indices[0]];
						mesh->vertex_positions[triangles_loaded*3+1]=mesh->vertices[triangle->vertex_indices[1]];
						mesh->vertex_positions[triangles_loaded*3+2]=mesh->vertices[triangle->vertex_indices[2]];

						mesh->vertex_normals[triangles_loaded*3+0]=mesh->normals[triangle->normal_indices[0]];
						mesh->vertex_normals[triangles_loaded*3+1]=mesh->normals[triangle->normal_indices[1]];
						mesh->vertex_normals[triangles_loaded*3+2]=mesh->normals[triangle->normal_indices[2]];

						mesh->vertex_tex_coords[triangles_loaded*3+0]=mesh->texture_coordinates[triangle->tex_coord_indices[0]];
						mesh->vertex_tex_coords[triangles_loaded*3+1]=mesh->texture_coordinates[triangle->tex_coord_indices[1]];
						mesh->vertex_tex_coords[triangles_loaded*3+2]=mesh->texture_coordinates[triangle->tex_coord_indices[2]];

						mesh->indices[triangles_loaded*3+0]=triangles_loaded*3+0;
						mesh->indices[triangles_loaded*3+1]=triangles_loaded*3+1;
						mesh->indices[triangles_loaded*3+2]=triangles_loaded*3+2;

						++triangles_loaded;
					}
	}
	fclose(fp);

	return mesh;
}
