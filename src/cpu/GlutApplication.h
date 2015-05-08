/***************************************************************************
glutapp.h  -  description
-------------------
begin                : Mo Feb 16 2004
copyright            : (C) 2003 by David Charypar
email                : charypar@inf.ethz.ch
***************************************************************************/


#ifndef GLUTAPP__H
#define GLUTAPP__H GLUTAPP__H

#include <map>



class GlutWindow;

typedef std::map<const int, GlutWindow *> IdToWindowMap;
typedef std::map<const GlutWindow *, int> WindowToIdMap;

/**
* Class to visualize about any simulation
*
*@author David Charypar
*/

class GlutApplication
{
private:
	static IdToWindowMap idToWindowMap;
	static WindowToIdMap windowToIdMap;


	/** drawing routine for the window.
	* this is where the "visualization" takes place
	*/
	static void displayCallback();

	/** This method is called every time the window changes its size.
	* Here you can react to the change in shape or size.
	* This is done usually by changing the projection matrix
	* and the viewport of openGL.
	*/
	static void reshapeCallback(int w, int h);

	/** This method is called every time a normal key is pressed on the keyboard.
	* Normal keys are keys that would produce a character in a normal editor.
	* Del, Backspace and ESC are NORMAL keys.
	*/
	static void keyboardCallback(unsigned char key, int x, int y);

	/** This method is called every time a special key is pressed on the keyboard.
	* Normal keys are keys that would not produce a character in a normal editor.
	* Del, Backspace and ESC are NORMAL keys.
	*/
	static void specialKeyboardCallback(int key, int x, int y);

	/** This method is called every time a mouseclick is performed*/
	static void mouseCallback(int button, int state, int x, int y);

	/** This method is called when the user drags something over the screen*/
	static void motionCallback(int x, int y);

	/** This method is called when the user moves the mouse over the screen*/	
	static void passiveMotionCallback(int x, int y);

	/**
	* Initializes the visualisation.
	*
	* Opens a window, initializes opengl, registeres all callback functions.
	*/
	void registerWindow(GlutWindow *win);

public:

	GlutApplication(const char *appName, GlutWindow *mainWindow);
	~GlutApplication();

	/**
	* Starts the application loop.
	*
	*/
	void run();

	/**
	* Registers a Subwindow
	*/
	void registerSubWindow(GlutWindow *parent, GlutWindow *subWin);

};


#endif


