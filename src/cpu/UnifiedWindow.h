#ifndef GPU_UNIFIED_WINDOW_H_
#define GPU_UNIFIED_WINDOW_H_

#include <cmath>
#include <vector>

#include "GlutWindow.h"
#include "globalPath.h"
#include "UnifiedIO.h"
#include "render_particles.h"

class GLTools
{
public:
	static void DrawSphere(const double x, const double y, const double z, const double radius)
	{
		// draw spheres -> quite slow
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(x, y, z);
		glutSolidSphere(radius, 6, 3);
		glPopMatrix();				
	}

	static void DrawPoint(const double x, const double y, const double z, const double radius)
	{
		// draw simple points of size 3 -> very fast
		glPointSize(radius);
		glBegin(GL_POINTS); 
		glVertex3f(x, y, z);
		glEnd();
	}



	static void SetLighting()
	{
		GLfloat mat_specular[] = {0.2, 0.2, 0.2, 1.0};
		GLfloat mat_shininess[] = {10.0};
		GLfloat light_position0[] = {-5.0, 5.0, 10.0, 0.0};      
		GLfloat light_position1[] = {5.0, 5.0, -10.0, 0.0};   

		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
		glLightfv(GL_LIGHT0, GL_POSITION, light_position0);  
		glLightfv(GL_LIGHT1, GL_POSITION, light_position1); 

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHT1);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_COLOR_MATERIAL);
	}

};

class UnifiedWindow: public GlutWindow
{
public:

	UnifiedWindow();
	~UnifiedWindow();

	virtual void Display();
	virtual void DisplayObjects();
	virtual void Reshape(const int width, const int height);
	virtual void Mouse(const int button, const int state, const int x, int y);
	virtual void Motion(const int x, const int y);
	virtual void Keyboard(const unsigned char key, const int x, const int y);
	virtual void Redisplay();

	void DrawDomain(const vmml::Vector3f& domain_min, const vmml::Vector3f& domain_max);

private:

	void RenderParticles(const int particleType, const float x, const float y, const float z, const float r);
	void RenderFromCPUSimulation();
	void RenderFromGPUSimulation();
	void UseVBOCPU();
	void UseWithoutVBO();

	void RenderWithVBO();
	void RenderWithoutVBO();

protected: 

	double fov_;
	double aspect_;
	int clickX_;
	int clickY_;
	bool left_button_;
	bool middle_button_;
	bool right_button_;
	double rotation_matrix_[16];
	double translation_[3];
	float sphere_size_factor_;
	DisplayType display_type;
	ParticleRenderer::DisplayMode display_mode_;

	int window_info_[4];
};

#endif	// GPU_UNIFIED_WINDOW_H_

