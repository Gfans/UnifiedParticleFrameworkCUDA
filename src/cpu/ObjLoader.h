#ifndef GPU_UNIFIED_OBJLOADER_H_
#define GPU_UNIFIED_OBJLOADER_H_ 

#include "ObjectLoader.h"
#include "Mesh.h"

/**
 * An ObjectLoader class that can handle triangulated meshes in the Wavefront
 * Object files (.obj) format.
 */
class ObjLoader : public ObjectLoader
{
	public:
		/**
		 * Singleton method.
		 */
		static ObjLoader& GetInstance();
	
	protected:
		/**
		 * Implements ObjectLoader::loadFaces(string fileName).
		 * The faces returned are of the class Polygon.
		 */
		virtual std::vector<Face*> LoadFaces(const std::string& file_name);
	
	private:
		/**
		 * Constructor. Use GetInstance() instead.
		 */
		ObjLoader();
		/**
		 * Turns the file into a mesh. 
		 * Adapted from the ObjLoader by Alexander Kraehenbuehl.
		 */
		Mesh* Load(const std::string& fileName) const; 
};

#endif	// GPU_UNIFIED_OBJLOADER_H_
