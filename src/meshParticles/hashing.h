// Include the STL hash_set implementation
#ifndef MDSIMHASH_H
#define MDSIMHASH_H


#ifdef __GNUC__

#ifdef __ICC
#include <ext/hash_set>
#include <ext/hash_map>

using stlport::hash;
using stlport::hash_set;
using stlport::hash_map;

#else

#include <ext/hash_set>
#include <ext/hash_map>

using __gnu_cxx::hash;
using __gnu_cxx::hash_set;
using __gnu_cxx::hash_map;

#endif

#elif defined WIN32

#define _DEFINE_DEPRECATED_HASH_CLASSES 0

#include <hash_set>
#include <hash_map>

/*
About stdext Namespace
http://msdn.microsoft.com/en-us/library/ek139e86(v=vs.100).aspx
*/

using std::hash;
using stdext::hash_set;
using stdext::hash_map;

#endif

#endif
