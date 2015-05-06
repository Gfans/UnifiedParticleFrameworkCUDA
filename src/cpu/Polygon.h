#ifndef GPU_UNIFIED_POLYGON_H_
#define GPU_UNIFIED_POLYGON_H_ 

#include <vector>
#include "Face.h"
#include "vmmlib/vector3.h"

/**
 * This class represents a polygonal Face.
 * A Polygon must at least have 3 vertices added, and their position must not be identical.
 * All vertices must be on one plane.
 */
class Polygon : public Face
{
public:
	Polygon();
	~Polygon();

	/**
	* Adds another vertex to the polygon at the position of 'vertex'.
	*/
	void AddVertex(const vmml::Vector3f& vertex);
	/**
	* Implements
	* Face::GetBoundaryBox(vmml::Vector3f* min_point, vmml::Vector3f* max_point).
	*/
	void GetBoundaryBox(vmml::Vector3f& min_point, vmml::Vector3f& max_point);
	/**
	* Implements
	* Face::DoesLineIntersect(vmml::Vector3f& start_pos, vmml::Vector3f& end_pos).
	*/
	bool DoesLineIntersect(const vmml::Vector3f& start_pos, const vmml::Vector3f& end_pos);
	/**
	* Implements
	* Face::DoesPointTouch(vmml::Vector3f& point).
	*/
	bool DoesPointTouch(vmml::Vector3f& point);

private:
	/**
	* Tests if point1 and point2 are on differing sides of the line from
	* lineStart to lineEnd.
	*/
	bool arePointsOnBothSidesOfTheLine(const vmml::Vector3f& lineStart, const vmml::Vector3f& lineEnd, const vmml::Vector3f& point1, const vmml::Vector3f& point2);
	/**
	* Tests if a point in the plane of the polygon is inside the polygon.
	*/
	bool isPointInPlaneInside(const vmml::Vector3f& point);

private:
	/**
	* Holds the vertices of this polygon.
	*/
	std::vector<vmml::Vector3f> vertices_;
	/**
	* Cached value for GetBoundaryBox
	*/
	vmml::Vector3f boundary_min_;
	/**
	* Cached value for GetBoundaryBox.
	*/
	vmml::Vector3f boundary_max_;
	/**
	* Normal of the polygon, cached for DoesLineIntersect.
	*/
	vmml::Vector3f normal_;

};

#endif	// GPU_UNIFIED_POLYGON_H_
