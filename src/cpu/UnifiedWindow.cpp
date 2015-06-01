#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>
#include <ctime>

#include "UnifiedWindow.h"
#include "UnifiedConstants.h"
#include "UnifiedPhysics.h"
#include "globalPath.h"
#include "global.h"
#include "render_particles.h"
#include "System/Timer.h"

UnifiedWindow::UnifiedWindow()
{
	fov_ = 20.0;
	aspect_ = 1.0;
	clickX_ = 0;
	clickY_ = 0;
	left_button_ = false;
	middle_button_ = false;
	right_button_ = false;

	for(int i = 0; i < 16; ++i) 
		rotation_matrix_[i] = 0.0;

	for(int i = 0; i < 16; i += 5) 
		rotation_matrix_[i] = 1.0;

	translation_[0] = 0.0;
	translation_[1] = 0.0;
	translation_[2] = 0.0;

	sphere_size_factor_ = 4.0f;

	display_type = SHOW_POINT; 

	display_mode_ = ParticleRenderer::PARTICLE_SPHERES;
}

UnifiedWindow::~UnifiedWindow()	
{

}

void UnifiedWindow::Display()
{

#ifdef SPH_PROFILING

	sr::sys::Timer total_time;
	total_time.Start();

#endif

	// main loop: call DoPhysics() which executes one iteration of the simulation
#ifdef USE_ASSERT
	assert(fc && myFluid);
#endif

	if(fc->animating || fc->one_animation_step)
	{
		myFluid->DoPhysics();
	}

#ifdef SPH_PROFILING

	sr::sys::Timer rendering_time;
	rendering_time.Start();

#endif

	DisplayObjects();

#ifdef SPH_PROFILING

	rendering_time.Stop();
	total_time.Stop();
	myFluid->time_counter_rendering_ += rendering_time.GetElapsedTime();
	myFluid->time_counter_total_ += total_time.GetElapsedTime();
	if (myFluid->frame_counter_ % SPH_PROFILING_FREQ == 0)
	{
		float averageElapsed = myFluid->time_counter_rendering_ / myFluid->frame_counter_;
		float averageElapsedTotal = myFluid->time_counter_total_ / myFluid->frame_counter_;
		std::cout << "  overall rendering          average: ";
		std::cout << std::fixed << std::setprecision(5) << averageElapsed << "s = ";
		std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsed << "fps" << std::endl;
		std::cout << "  total time          average: ";
		std::cout << std::fixed << std::setprecision(5) << averageElapsedTotal << "s = ";
		std::cout << std::setprecision(2) << std::setw(7) << 1.0f / averageElapsedTotal << "fps" << std::endl;
	}

#endif

	fc->one_animation_step = false;

	glutSwapBuffers();

#ifdef USE_FFMPEG

	glReadPixels(0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE, myFluid->buffer_);

	fwrite(myFluid->buffer_, sizeof(int)*800*600, 1, myFluid->ffmpeg_);

#endif
}

void UnifiedWindow::DisplayObjects()
{
	srand ( time ( 0x0 ) );

	double fovy;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//lighting
	GLTools::SetLighting();

	if(aspect_>1.0)
		fovy = fov_;
	else
		fovy = fov_ / aspect_;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect_, 3.0 * fc->sizeFactor, 100.0 * fc->sizeFactor);	

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(0, fc->virtualBoundingBox.getMax().y + fc->sizeFactor, fc->virtualBoundingBox.getMax().z + 7.0 * fc->sizeFactor, 0, 0.5f * fc->virtualBoundingBox.getMax().y, 0, 0, 1, 0);

	// do translation
	glTranslatef(translation_[0], translation_[1], translation_[2]);
	// rotate around origin
	//glRotatef(45, 0.0, 1.0, 0.0);
	glMultMatrixd(rotation_matrix_);

	// draw the border of the scene
	//glColor3f(0, 0, 0);
	//myFluid->DrawBox(fc->virtualBoundingBox);

	glColor3f(1.0, 0.0, 0.0);
	myFluid->DrawBox(fc->realBoxContainer);

	//glColor3f ( 0.1, 0.1, 0.2 );
	//myFluid->drawGroundGrid(fc->realBoxContainer);

	// draw particles	
	if(fc->drawParticles == 0)
		return;

#ifdef USE_CUDA
	RenderFromGPUSimulation();
#else
	RenderFromCPUSimulation();
#endif

}

void UnifiedWindow::Reshape(const int width, const int height)	
{
	aspect_ = static_cast<double>(width) / static_cast<double>(height);

	double fovy;
	if(aspect_>1.0)
		fovy = fov_;
	else
		fovy = fov_ / aspect_;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect_, 3.0 * fc->sizeFactor, 100.0 * fc->sizeFactor);	

	glMatrixMode(GL_MODELVIEW);

	glViewport(0,0,width,height);

#if defined(USE_VBO_CUDA) || defined(USE_VBO_CPU)
	ParticleRenderer *render = myFluid->renderer();

	if (render)
	{
		render->setWindowSize(width, height);
		render->setFOV(60.0);
	}
#endif

	Redisplay();
}

void UnifiedWindow::Mouse(const int button, const int state, const int x, int y)
{
	clickX_ = x;
	clickY_ = y;
	if(state == GLUT_DOWN)
	{
		switch(button)
		{
		case GLUT_LEFT_BUTTON:      left_button_ = true; break;
		case GLUT_MIDDLE_BUTTON:	middle_button_ = true; break;
		case GLUT_RIGHT_BUTTON:     right_button_ = true; break;
		}
	}
	else
	{
		switch(button)
		{
		case GLUT_LEFT_BUTTON:      left_button_ = false; break;
		case GLUT_MIDDLE_BUTTON:	middle_button_ = false; break;
		case GLUT_RIGHT_BUTTON:     right_button_ = false; break;
		}
	}

	glGetIntegerv(GL_VIEWPORT, (GLint*)window_info_);
}

void UnifiedWindow::Motion(const int x, const int y)
{
	if(left_button_ == true)
	{
		// we want to rotate           
		int deltaX = x - clickX_;
		int deltaY = y - clickY_;

		clickX_ = x;
		clickY_ = y;

		if ((deltaX == 0) && (deltaY == 0))
			return;

		double axisX = deltaY;
		double axisY = deltaX;
		double axisZ = 0.0;

		glGetIntegerv(GL_VIEWPORT, (GLint*)window_info_);

		double angle = 180.0 * sqrt(static_cast<double>(deltaX * deltaX + deltaY * deltaY)) / static_cast<double>(window_info_[2] + 1);

		// -> calculate axis in world coordinates 
		double wx = rotation_matrix_[0] * axisX + rotation_matrix_[1] * axisY + rotation_matrix_[ 2]  * axisZ;
		double wy = rotation_matrix_[4] * axisX + rotation_matrix_[5] * axisY + rotation_matrix_[ 6] * axisZ;
		double wz = rotation_matrix_[8] * axisX + rotation_matrix_[9] * axisY + rotation_matrix_[10] * axisZ;

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		glLoadMatrixd(rotation_matrix_);
		glRotated(angle, wx, wy, wz);
		glGetDoublev(GL_MODELVIEW_MATRIX, rotation_matrix_);
		glPopMatrix();

	}

	if(middle_button_ == true)
	{
		int deltaX = x - clickX_;
		int deltaY = y - clickY_;

		clickX_ = x;
		clickY_ = y;

		glGetIntegerv(GL_VIEWPORT, (GLint*)window_info_);

		fov_ *= static_cast<double>(window_info_[3] - deltaY) / window_info_[3];
	}

	if(right_button_ == true)
	{
		int deltaX = x - clickX_;
		int deltaY = y - clickY_;

		clickX_ = x;
		clickY_ = y;

		//translation is always camera system
		double dx = deltaX;
		double dy = -deltaY;
		double dz = 0.0;

		glGetIntegerv(GL_VIEWPORT, (GLint*)window_info_);

		dx *= fc->sizeFactor * 0.2 * fov_/window_info_[3];
		dy *= fc->sizeFactor * 0.2 * fov_/window_info_[3];
		dz *= fc->sizeFactor * 0.2 * fov_/window_info_[3];

		translation_[0] += dx;
		translation_[1] += dy;
		translation_[2] += dz;
	}

	Redisplay(); 
}

void UnifiedWindow::Keyboard(const unsigned char key, const int x, const int y)
{
	std::cout << "key: " << key << " / ";
	int num;
	switch(key)
	{
	case 'a':
		// start and stop animation			
		fc->animating = !fc->animating;
		std::cout << "animating: " << ((fc->animating)? "true": "false") << std::endl;
		break;
	case 'd':
		fc->displayEnabled = !fc->displayEnabled;
		break;
	case 'o':
		//write particle positions into file   
		if(fc->saveParticles)
			fc->saveParticles = false;
		else
			fc->saveParticles = true;
		break;
	case 'Q':
		exit(0);
		break;
	case 's' :
		// do one animation step
		fc->one_animation_step = true;
		std::cout << "One animation step" << std::endl;
		break;
	case '0':
		// draw nothing
		fc->drawParticles = (fc->drawParticles+1) % 2;//!fc->drawNothing;
		std::cout << "draw particles: " << fc->drawParticles << std::endl;
		break;	
	case '-':
		sphere_size_factor_ *= 0.9;			
		break;
	case 'r' :
		// restart
		{
			delete myFluid;
			myFluid = new UnifiedPhysics(fc); 
			break;
		}
	case '+':
		sphere_size_factor_ *= 1.1;			
		break;
	case 'z':	// display rigid particle
		fc->drawRigidParticle = !fc->drawRigidParticle;
		break;
	case 'x':	// display soft/deformable particle
		fc->drawSoftParticle = !fc->drawSoftParticle;
		break;
	case 'c':	// display liquid particle
		fc->drawLiquidParticle = !fc->drawLiquidParticle;
		break;
	case 'v':	// display granular particle
		fc->drawGranularParticle = !fc->drawGranularParticle;
		break;
	case 'b':	// display cloth particle
		fc->drawClothParticle = !fc->drawClothParticle;
		break;
	case 'n':	// display smoke particle
		fc->drawSmokeParticle = !fc->drawSmokeParticle;
		break;
	case 'm':	// display frozen particle
		fc->drawFrozenParticle = !fc->drawFrozenParticle;
		break;
	case 'p':
#if defined(USE_VBO_CUDA) || defined(USE_VBO_CPU)
		display_mode_ = (ParticleRenderer::DisplayMode)
			((display_mode_ + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
#else
		if (display_type == SHOW_SPHERE) {
			display_type = SHOW_POINT;
		}else {
			display_type = SHOW_SPHERE;
		}
#endif
		break;
	}
	glutPostRedisplay();	
}

void UnifiedWindow::Redisplay()
{
	glutPostRedisplay();
}

void UnifiedWindow::DrawDomain(const vmml::Vector3f& domain_min, const vmml::Vector3f& domain_max)	
{	
	glColor3f ( 0.0, 0.0, 1.0 );
	/*
	glBegin ( GL_LINES );
	glVertex3f ( domain_min.x, domain_min.y, domain_min.z );	
	glVertex3f ( domain_max.x, domain_min.y, domain_min.z );
	glVertex3f ( domain_min.x, domain_max.y, domain_min.z );	
	glVertex3f ( domain_max.x, domain_max.y, domain_min.z );
	glVertex3f ( domain_min.x, domain_min.y, domain_min.z );	
	glVertex3f ( domain_min.x, domain_max.y, domain_min.z );
	glVertex3f ( domain_max.x, domain_min.y, domain_min.z );	
	glVertex3f ( domain_max.x, domain_max.y, domain_min.z );
	glEnd ();
	*/

	// draw the domain

	/*
	/8------------ /7
	/|             / |
	/	|			 /  |
	/5-|-----------6	|
	|  |			|
	|	|			|	|
	|	|			|	|
	|	4-----------|---3
	|	/			|  /
	| /			| /
	|1 ----------- 2	

	*/
	/*
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	*/
	// ground (1234)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	//ceil(5678)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	//left face (14,58,15,48)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glEnd();

	//right face (23,67,26,37)
	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	//back face(43,78,37,48)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glEnd();

	//front face(12,56,15,26)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();
}

void UnifiedWindow::RenderParticles(const int particleType, const float x, const float y, const float z, const float r)
{
	// 7 colors in rainbow
	// Red(1.0, 0.0, 0.0), Orange(1.0, 0.647, 0.0), Yellow(1.0, 1.0, 0.0), Green(0, 1.0, 0.0), Blue(0.0, 0.0, 1.0), Indigo(0.294, 0.0,0.509), Violet(0.933, 0.509, 0.933) 

	if (particleType == LIQUID_PARTICLE)
	{
		glColor3f(0.0, 0.0, 1.0);			//Blue
	}
	else if (particleType == SOFT_PARTICLE)
	{
		glColor3f(1.0, 0.647, 0.0);			//Orange
	} 
	else if (particleType == RIGID_PARTICLE)
	{
		glColor3f(0, 1.0, 0.0);				//Green
	} 
	else if (particleType == GRANULAR_PARTICLE)
	{
		glColor3f(0.294, 0.0,0.509);		//Indigo
	} 
	else if (particleType == CLOTH_PARTICLE)
	{
		glColor3f(0.933, 0.509, 0.933);		//Violet
	}
	else if (particleType == SMOKE_PARTICLE)
	{
		glColor3f(1.0, 1.0, 1.0);			//Gray
	} 
	else if (particleType == FROZEN_PARTICLE)
	{	
		// for debugging purpose : currently we just display frozen particles as points
		glColor3f(1.0, 1.0, 0.0);			//Yellow
		//GLTools::DrawPoint(x, y, z, 50 * r);
		//return; 		
	}

	if (SHOW_SPHERE == display_type)
		GLTools::DrawSphere(x, y, z, r);
	else if(SHOW_POINT == display_type)
		GLTools::DrawPoint(x, y, z, r);

}

void UnifiedWindow::RenderFromCPUSimulation()
{
	// render particles from cpu simulation
#ifdef USE_VBO_CPU	

	UseVBOCPU();

#else

	UseWithoutVBO();

#endif // end #ifdef USE_VBO_CPU
}

void UnifiedWindow::RenderFromGPUSimulation()
{
	// render particles from GPU simulation
#ifdef USE_VBO_CUDA

	RenderWithVBO();

#else

	RenderWithoutVBO();

#endif
}

void UnifiedWindow::RenderWithVBO()
{
	ParticleRenderer *myRenderer = myFluid->renderer();
	if ( myRenderer && fc->displayEnabled)
	{
		myRenderer->display(display_mode_);
	}
}

void UnifiedWindow::RenderWithoutVBO()
{
	//float r = SPHERE_RADIUS * fc->particleSpacing * sphere_size_factor_;
	float r = fc->particleRadius;
	int size = 3;
	const uint numParticles = myFluid->particles().size();
	for (int i = 0; i < numParticles; ++i)
	{	
		float x = myFluid->particle_info_for_rendering().p_pos_zindex[4*i];
		float y = myFluid->particle_info_for_rendering().p_pos_zindex[4*i+1];
		float z = myFluid->particle_info_for_rendering().p_pos_zindex[4*i+2];

		int type = myFluid->particle_info_for_rendering().p_type[i];

		/*
		float t = i / (float) numParticles;
		GLfloat color_vec[3];
		myFluid->colorRamp(t, color_vec);
		glColor3fv(color_vec);
		*/

		if (type == LIQUID_PARTICLE)
		{
			if (!fc->drawLiquidParticle)
			{
				continue;
			}

			glColor3f(0.0, 0.0, 1.0); 

		}
		else if (type == FROZEN_PARTICLE)
		{
			if (!fc->drawFrozenParticle)
			{
				continue;
			}

			glColor3f(0.0, 1.0, 0.0);
		}
		else if (type == RIGID_PARTICLE)
		{
			if (!fc->drawRigidParticle)
			{
				continue;
			}

			glColor3f(1.0, 0.0, 0.0);
		}

		if (SHOW_SPHERE == display_type)
			GLTools::DrawSphere(x, y, z, r);
		else if(SHOW_POINT == display_type)
			GLTools::DrawPoint(x, y, z, r);
	}
}

void UnifiedWindow::UseVBOCPU()
{
	/************************************************************************/
	/*           using vbo for CPU simulation                               */
	/************************************************************************/
#ifdef  HIDE_FROZEN_PARTICLES

	// transfer updated data to GPU before soring
	myFluid->UpdatePositionVBOWithoutFrozenParticles();

	if (myFluid->renderer())
	{
		myFluid->renderer()->setVertexBuffer(myFluid->GetVBOCpu(), myFluid->GetNumNonfrozenParticles());
	}

#else

	// transfer updated data to GPU before soring
	myFluid->UpdatePositionVBO();

	if (myFluid->renderer())
	{
		myFluid->renderer()->setVertexBuffer(myFluid->GetVBOCpu(), myFluid->particles().size());
	}

#endif // #ifdef  HIDE_FROZEN_PARTICLES

	RenderWithVBO();
}

void UnifiedWindow::UseWithoutVBO()
{
	/************************************************************************/
	/*            CPU simulation without vbo rendering                       */
	/************************************************************************/
	float x, y, z;	
	//float r = SPHERE_RADIUS * fc->particleSpacing * sphere_size_factor_;
	float r = fc->particleRadius;
	int size = 3;
	const uint numParticles = myFluid->particles().size();
	for (int i = 0; i < numParticles; ++i)
	{	
		UnifiedParticle &p = myFluid->particles()[i];
		x = myFluid->particles()[i].position_.x;
		y = myFluid->particles()[i].position_.y;
		z = myFluid->particles()[i].position_.z;		

		RenderParticles(p.type_, x, y, z, r);
	}
}