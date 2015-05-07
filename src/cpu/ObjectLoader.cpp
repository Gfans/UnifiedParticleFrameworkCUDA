#include "ObjectLoader.h"

#include <iostream>

#include "vmmlib/vector3.h"

//--------------------------------------------------------------------
std::vector<UnifiedParticle>* ObjectLoader::LoadParticles(const std::string& file_name, const float scale_x, const float scale_y, const float scale_z, const float spacing, UnifiedParticle& particle_template)
{
	std::vector<UnifiedParticle>* result = new std::vector<UnifiedParticle>();

	std::vector<Face*> faces;
	try
	{
		faces = LoadFaces(file_name);
	}
	catch(FileNotFound)
	{
		std::cout << file_name << " could not be found!" << std::endl;
	}
	catch(FileError)
	{
		std::cout << "There was an error loading " << file_name << "." << std::endl;
	}
	std::vector<Face*>::iterator end_iter = faces.end();
	if(faces.begin() == end_iter) // true for empty files/after catching an error
	{
		return result;
	}

	// find the smallest and greatest values of all coordinates
	vmml::Vector3f min_vec, max_vec;
	faces[0]->GetBoundaryBox(min_vec, max_vec);
	for(std::vector<Face*>::iterator f = faces.begin(); f != end_iter; ++f)
	{
		Face* face = *f;

		vmml::Vector3f current_min, current_max;
		face->GetBoundaryBox(current_min, current_max);
		for(int i=0;i<3;++i)
		{
			min_vec[i] = std::min(min_vec[i],current_min[i]);
			max_vec[i] = std::max(max_vec[i],current_max[i]);
		}
	}

	// place the particles
	float x_steps = (max_vec[0]-min_vec[0])*scale_x/spacing;
	float y_steps = (max_vec[1]-min_vec[1])*scale_y/spacing;
	float z_steps = (max_vec[2]-min_vec[2])*scale_z/spacing;
	vmml::Vector3f outside = min_vec - spacing;
	vmml::Vector3f current_pos = min_vec; // position to evaluate in 3d model coords
	int particles_tried=0; // DEL
	for(int x=0; x <= x_steps; ++x)
	{
		current_pos[1] = min_vec[1]+(x%2)*spacing/(2*scale_y);
		for(int y=0; y <= y_steps; ++y)
		{
			current_pos[2] = min_vec[2]+(x%2)*spacing/(2*scale_z);
			for(int z=0; z <= z_steps; ++z)
			{
				bool should_add = false;
				for(std::vector<Face*>::iterator f = faces.begin(); f != end_iter; ++f)
				{
					Face* face = *f;

					if(face->DoesLineIntersect(outside,current_pos)) // add particle if number of intersection isn't odd
					{
						should_add = !should_add;
					}
					else if(face->DoesPointTouch(current_pos)) // add particles on boundaries
					{
						should_add = true;
						break;
					}
				}
				if(should_add)
				{
					particle_template.position.set((current_pos[0]-min_vec[0])*scale_x, (current_pos[1]-min_vec[1])*scale_y, (current_pos[2]-min_vec[2])*scale_z);
					result->push_back(particle_template);
				}
				++particles_tried; // DEL
				current_pos[2] += spacing/scale_z;
			}
			current_pos[1] += spacing/scale_y;
		}
		current_pos[0] += spacing/scale_x;
	}
	std::cout << "faces: " << faces.size() << " particles: " << result->size() << " tried: " << particles_tried << std::endl; // DEL
	// get rid of the faces
	for(std::vector<Face*>::iterator f = faces.begin(); f != end_iter; ++f)
	{
		Face* face = *f;
		delete face;
	}
	return result;
}
