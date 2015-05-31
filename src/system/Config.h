#ifndef GPU_UNIFIED_CONFIG_H
#define GPU_UNIFIED_CONFIG_H

// compiler & architecture
#ifdef __GNUC__
#  define SPH_GCC
#  if defined(_X86) || defined(__i386) || defined(i386) || defined (__amd64)
#    define SPH_X86
#    define SPH_ALIGN __attribute__((aligned (16)))
#  endif
#elif defined(_MSC_VER)
#  define SPH_MSVC
#  if defined(_M_IX86) || defined(_M_X64) || defined(_M_AMD64)
#    define SPH_X86
#    define SPH_ALIGN __declspec(align(16))
#  endif
#endif

// os
#if defined(__APPLE__) || defined(MACOSX)
#  define SPH_MAC
#elif defined(__linux)
#  define SPH_LINUX
#elif defined(_WIN32)
#  define SPH_WIN32
#endif

// byte order
#if defined(SPH_MAC) || defined(SPH_LINUX)
#  ifdef SPH_MAC
#    include <machine/endian.h>
#  elif defined(SPH_LINUX)
#    include <endian.h>
#  endif
#  ifdef BYTE_ORDER
#    if BYTE_ORDER == BIG_ENDIAN
#      define SPH_BIG_ENDIAN
#    elif BYTE_ORDER == LITTLE_ENDIAN
#      define SPH_LITTLE_ENDIAN
#    endif
#  endif
#elif defined(SPH_WIN32)
#  define SPH_LITTLE_ENDIAN
#endif

#endif
