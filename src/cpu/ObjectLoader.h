#ifndef GPU_UNIFIED_OBJECT_LOADER_H_
#define GPU_UNIFIED_OBJECT_LOADER_H_ 

#include <string>
#include <vector>
#include "Face.h"
#include "UnifiedParticle.h"

/**
* Subclasses of the ObjectLoader can turn a 3D model into a vector of 
* particles. The ObjectLoader class itself creates the particles using a
* vector of faces, which has to be provided by the subclass.
*/
class ObjectLoader
{
public:
	/**
	* Returns the object in fileName. A new vector is allocated for this
	* purpose that must be deleted by the caller. 
	* The object is scaled by scale_x, scale_y, scale_z.
	* The particles have the same properties as particle_template.
	* spacing determines the distance between the particles.
	*/
	std::vector<UnifiedParticle>* LoadParticles(const std::string& file_name, const float scale_x, const float scale_y, const float scale_z, const float spacing, UnifiedParticle& particle_template);

protected:
	/**
	* Turns a file into new Face objects
	*/
	virtual std::vector<Face*> LoadFaces(const std::string& file_name) = 0;
	/**
	* Thrown if a file doesn't exist
	*/
	struct FileNotFound
	{
		/**
		* Constructor. Takes the name of the non-existent file.
		*/
		FileNotFound(const std::string& name) : file(name){}
		/**
		* Name of the non-existent file.
		*/
		std::string file;
	};
	/**
	* Thrown if a file was corrupted
	*/
	struct FileError
	{
		/**
		* Constructor. Takes the name of the corrupt file.
		*/
		FileError(const std::string& name) : file(name){}
		/**
		* Name of the corrupt file.
		*/
		std::string file;
	};
};

#endif	// GPU_UNIFIED_OBJECT_LOADER_H_
