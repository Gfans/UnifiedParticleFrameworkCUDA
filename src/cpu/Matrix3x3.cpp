#include "Matrix3x3.h"

//When inverting a matrix, it will be considered non-invertible if the determinant 
//is below this number:
#define DETERMINANT_ZERO_BOUNDARY 1e-20

//--------------------------------------------------------------------
Matrix3x3::Matrix3x3()
	//--------------------------------------------------------------------
{

}

//--------------------------------------------------------------------
Matrix3x3 Matrix3x3::operator*(const Matrix3x3& matrix)
	//--------------------------------------------------------------------
{
	Matrix3x3 result;
	result.SetZero();
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			for(int k=0;k<3;++k)
			{
				result.elements[i][j] += elements[i][k]*matrix.elements[k][j];
			}
		}
	}
	return result;
}

//--------------------------------------------------------------------
vmml::Vector3f Matrix3x3::operator*(vmml::Vector3f& vector)
	//--------------------------------------------------------------------
{
	vmml::Vector3f result;
	result.set(0.0f, 0.0f, 0.0f);
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result[i] += elements[i][j]*vector[j];
		}
	}
	return result;
}

//--------------------------------------------------------------------
Matrix3x3 Matrix3x3::operator*(float factor)
	//--------------------------------------------------------------------
{
	Matrix3x3 result;
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result.elements[i][j] = elements[i][j] * factor;
		}
	}
	return result;
}

//--------------------------------------------------------------------
Matrix3x3 Matrix3x3::operator+(const Matrix3x3& summand)
	//--------------------------------------------------------------------
{
	Matrix3x3 result;
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result.elements[i][j] = elements[i][j]+summand.elements[i][j];
		}
	}
	return result;
}

//--------------------------------------------------------------------
void Matrix3x3::SetZero()
{
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			elements[i][j] = 0.0f;
		}
	}
}

//--------------------------------------------------------------------
Matrix3x3 Matrix3x3::GetTransposedMatrix()
{
	Matrix3x3 result;
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result.elements[j][i]=elements[i][j];
		}
	}
	return result;
}

//--------------------------------------------------------------------
Matrix3x3 Matrix3x3::UnitMatrix()
{
	Matrix3x3 result;
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result.elements[i][j]=0.0f;
		}
	}
	result.elements[0][0]=result.elements[1][1]=result.elements[2][2]=1.0f;
	return result;
}

//--------------------------------------------------------------------
bool Matrix3x3::CalculateInverse(const Matrix3x3& matrix, Matrix3x3& result)
{
	// calculate det(matrix)
	float det =	matrix.elements[0][0]*matrix.elements[1][1]*matrix.elements[2][2] - matrix.elements[2][0]*matrix.elements[1][1]*matrix.elements[0][2]
	+matrix.elements[0][1]*matrix.elements[1][2]*matrix.elements[2][0] - matrix.elements[2][1]*matrix.elements[1][2]*matrix.elements[0][0]
	+matrix.elements[0][2]*matrix.elements[1][0]*matrix.elements[2][1] - matrix.elements[2][2]*matrix.elements[1][0]*matrix.elements[0][1];
	if(det<DETERMINANT_ZERO_BOUNDARY && det>-DETERMINANT_ZERO_BOUNDARY)
	{
		return false;
	}
	float detFactor = 1/det;

	// calculate inverse by calculating the adjoint ("Komplementäre Matrize") and dividing by det
	for(int i=0; i<3; ++i)
	{
		for(int j=0; j<3; ++j)
		{
			// find 2x2 matrix not containing row i & column j
			float elementsOfSubmatrix[4];
			int elementIndex=0;
			for(int k=0; k<3; ++k)
			{
				for(int l=0; l<3; ++l)
				{
					if(i!=k && j!=l)
					{
						elementsOfSubmatrix[elementIndex] = matrix.elements[k][l];
						++elementIndex;
					}
				}
			}
			result.elements[j][i] = elementsOfSubmatrix[0]*elementsOfSubmatrix[3]-elementsOfSubmatrix[2]*elementsOfSubmatrix[1];
			// multiply by (-1)^(i+j) to get the adjoint, divide by det to get the inverse:
			result.elements[j][i] *= ((i+j)&0x1) ? -detFactor : detFactor;
		}
	}
	return true;
}
