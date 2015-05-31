#ifndef LIBBELL_H
#define LIBBELL_H

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifndef mkdir
#ifdef WIN32
#include <direct.h>
#define mkdir(a,b) _mkdir(a)
#endif
#endif

#ifndef uint
typedef unsigned int uint;
#endif

#include <assert.h>

#endif
