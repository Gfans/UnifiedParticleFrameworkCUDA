#ifndef GPU_UNIFIED_MATRIX_3X3_H_
#define GPU_UNIFIED_MATRIX_3X3_H_

#include "vmmlib/vector3.h"

/**
 * Represents a matrix of size 3x3.
 */
class Matrix3x3
{
	public:
		/**
		 * Constructor. The elements are not set.
		 */
		Matrix3x3();
		/**
		 * Multiplies 'matrix' from the right.
		 */
		Matrix3x3 operator*(const Matrix3x3& matrix);
		/**
		 * Matrix-vector multiplication.
		 */
		vmml::Vector3f operator*(vmml::Vector3f& vector);
		/**
		 * Multiplies all elements with factor.
		 */
		Matrix3x3 operator*(float factor);
		/**
		 * Matrix addition.
		 */
		Matrix3x3 operator+(const Matrix3x3& summand);
		/**
		 * Sets all elements to zero.
		 */
		void SetZero();
		/**
		 * Returns the transpose of this matrix.
		 */
		Matrix3x3 GetTransposedMatrix();
		/**
		 * Returns an unit matrix.
		 */
		static Matrix3x3 UnitMatrix();
		/**
		 * Calculates the inverse of matrix and stores it in result. If the
		 * matrix could not be inverted, result is not changed and false will be
		 * returned.
		 */
		static bool CalculateInverse(const Matrix3x3& matrix, Matrix3x3& result);
		/**
		 * Contains the elements of this matrix.
		 */
		float elements[3][3];
};

#endif	// GPU_UNIFIED_MATRIX_3X3_H_
