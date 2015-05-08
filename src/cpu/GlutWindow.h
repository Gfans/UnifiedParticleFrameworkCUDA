#ifndef GPU_UNIFIED_GLUT_WINDOW_H_
#define GPU_UNIFIED_GLUT_WINDOW_H_

//--------------------------------------------------------------------
class GlutWindow
	//--------------------------------------------------------------------
{
public:

	GlutWindow();
	~GlutWindow();

	virtual void display();
	virtual void reshape(int width, int height);
	virtual void keyboard(unsigned char key, int x, int y);
	virtual void specialKeyboard(int key, int x, int y);
	virtual void mouse(int button, int state, int x, int y);
	virtual void motion(int x, int y);
	virtual void passiveMotion(int x, int y);

};

#endif	// GPU_UNIFIED_GLUT_WINDOW_H_

