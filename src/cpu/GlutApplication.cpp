/***************************************************************************
glutapp.cpp  description
-------------------
begin                : Mo Feb 16 2004
copyright            : (C) 2003 by David Charypar
email                : charypar@inf.ethz.ch
***************************************************************************/


#include "GlutApplication.h"
#include "GlutWindow.h"
#include "globalPath.h"
#include "global.h"
#include <iostream>
#include <new>
#include <cstring>


IdToWindowMap GlutApplication::idToWindowMap;
WindowToIdMap GlutApplication::windowToIdMap;


void GlutApplication::displayCallback()
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows display() method
	idToWindowMap[currentWindowId]->display();
}

void GlutApplication::reshapeCallback(int w, int h)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows reshape() method
	idToWindowMap[currentWindowId]->reshape(w, h);
}


void GlutApplication::keyboardCallback(unsigned char key, int x, int y)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows keyboard() method
	idToWindowMap[currentWindowId]->keyboard(key, x, y);
}

void GlutApplication::specialKeyboardCallback(int key, int x, int y)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows specialKeyboard(key, x, y) method
	idToWindowMap[currentWindowId]->specialKeyboard(key, x, y);
}

void GlutApplication::mouseCallback(int button, int state, int x, int y)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows mouse(button, state, x, y) method
	idToWindowMap[currentWindowId]->mouse(button, state, x, y);
}

void GlutApplication::motionCallback(int x, int y)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows motion() method
	idToWindowMap[currentWindowId]->motion(x, y);
}

void GlutApplication::passiveMotionCallback(int x, int y)
{
	// get current Window id
	int currentWindowId = glutGetWindow();

	// call the corresponding windows passiveMotion() method
	idToWindowMap[currentWindowId]->passiveMotion(x, y);
}

GlutApplication::GlutApplication(const char *appName, GlutWindow *mainWindow)
{
	// dummy variables to call init
	int argCounter = 1;
	char *command[1];
	char appNameCopy[1000];
	strcpy(appNameCopy, appName);

	command[0] = appNameCopy;

	glutInit(&argCounter, command);
	//		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

	registerWindow(mainWindow);
	glutSetWindowTitle(appName);

	const GLubyte* name = glGetString(GL_VENDOR); 
	const GLubyte* renderer = glGetString(GL_RENDERER); 
	const GLubyte* OpenGLVersion =glGetString(GL_VERSION); 
	printf("----------------------OpenGL Information----------------------\n");
	printf("Rendering GPU£º%s\n", renderer);
	printf("OpenGL version£º%s\n",OpenGLVersion );

	glewExperimental = true;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cout << "glewInitError!!!" << std::endl;
		exit(1);
	}

	std::cout << "GLEW initialized OK!!!" << std::endl;
	std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl << std::endl;

	if (!glewIsSupported("GL_VERSION_2_0"))
	{
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(EXIT_FAILURE);
	}

#if defined (_WIN32)

	if (wglewIsSupported("WGL_EXT_swap_control"))
	{
		// disable vertical sync
		wglSwapIntervalEXT(0);
	}

	glEnable(GL_DEPTH_TEST);
	// background color
	glClearColor(0.9f, 0.9f, 0.9f, 0.0f);

#endif

}

GlutApplication::~GlutApplication()
{
	std::cerr << "hallo" << std::endl;
	while(!idToWindowMap.empty())
	{
		IdToWindowMap::iterator end=idToWindowMap.end();
		std::cerr << "deleting window with id " << end->first << std::endl;
		delete end->second;
		idToWindowMap.erase(end);
		std::cerr << "done. " << std::endl;
	}
}


void GlutApplication::run()
{
	glutMainLoop();
}

void GlutApplication::registerWindow(GlutWindow *win)
{
	int id = glutCreateWindow("GLUT Window");
	//cerr << "Id is " << id << endl;

	// store relation between map and window
	idToWindowMap[id] = win;
	windowToIdMap[win] = id;

	// register callbacks
	glutDisplayFunc(displayCallback);
	glutReshapeFunc(reshapeCallback);
	glutKeyboardFunc(keyboardCallback);
	glutSpecialFunc(specialKeyboardCallback);
	glutMouseFunc(mouseCallback);
	glutMotionFunc(motionCallback);
	glutPassiveMotionFunc(passiveMotionCallback);


}

void GlutApplication::registerSubWindow(GlutWindow *parent, GlutWindow *subWin)
{
	// find id of parent
	int parentId = windowToIdMap[parent];	

	int id = glutCreateSubWindow(parentId, 40, 30, 100, 150);
	//cerr << "Id is " << id << endl;

	// store relation between map and window
	idToWindowMap[id] = subWin;
	windowToIdMap[subWin] = id;


	// register callbacks
	glutDisplayFunc(displayCallback);
	glutReshapeFunc(reshapeCallback);
	glutKeyboardFunc(keyboardCallback);
	glutSpecialFunc(specialKeyboardCallback);
	glutMouseFunc(mouseCallback);
	glutMotionFunc(motionCallback);
	glutPassiveMotionFunc(passiveMotionCallback);

}



