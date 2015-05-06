#ifndef GPU_UNIFIED_FACE_H_
#define GPU_UNIFIED_FACE_H_

#include "vmmlib/vector3.h"

/**
 * Abstract superclass for all types of faces that can be used when loading a
 * ParticleObject from a 3D model.
 */
class Face
{
	public:
		virtual ~Face();
		/**
		 * Sets min and max so that an axis-aligned bounding box around this 
		 * face has corners on min and max.
		 */
		virtual void GetBoundaryBox(vmml::Vector3f& min_point, vmml::Vector3f& max_point) = 0;
		/**
		 * Returns true if the line between start_pos and end_pos intersects this 
		 * Face. Just touching is not considered intersecting. Freeform faces 
		 * like spheres should only register an intersection if the 
		 * inside/outside state of the endpoints is different.
		 */
		virtual bool DoesLineIntersect(const vmml::Vector3f& start_pos, const vmml::Vector3f& end_pos) = 0;
		/**
		 * Returns true if point lies on this Face.
		 */
		virtual bool DoesPointTouch(vmml::Vector3f& point) = 0;
};

#endif	// GPU_UNIFIED_FACE_H_
