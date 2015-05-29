#ifndef GPU_UNIFIED_GLUT_WINDOW_H_
#define GPU_UNIFIED_GLUT_WINDOW_H_

//--------------------------------------------------------------------
class GlutWindow
	//--------------------------------------------------------------------
{
public:

	GlutWindow();
	~GlutWindow();

	virtual void Display();
	virtual void Reshape(const int width, const int height);
	virtual void Keyboard(const unsigned char key, const int x, const int y);
	virtual void specialKeyboard(int key, int x, int y);
	virtual void Mouse(const int button, const int state, const int x, int y);
	virtual void Motion(const int x, const int y);
	virtual void passiveMotion(int x, int y);

};

#endif	// GPU_UNIFIED_GLUT_WINDOW_H_

