#ifndef SPATIALMAP_H
#define SPATIALMAP_H

#include <vector>
#include "hashing.h"
#include "cell.h"


template <class T> class CellContents : public std::vector< T >{
 public:
  void Add(const T& t){
    CellContents::push_back(t);
  }
  
  void Remove(const T& t){
    for(typename CellContents::iterator i = CellContents::begin(); i != CellContents::end(); i++){      
      if(*i == t){
	*i = std::vector<T>::back();
	std::vector<T>::pop_back();
	return;
      }
    }
  }


  bool Contains(const T& t) const{
    for(typename CellContents::const_iterator i = CellContents::begin(); i != CellContents::end(); i++){
      if(*i == t){
	return true;
      }
    }
    return false;
  }

  bool Empty() const{
    return CellContents::empty();
  }

  void ToString() const{
    for(typename CellContents::const_iterator i = CellContents::begin(); i != CellContents::end(); i++){
      std::cout << *i << std::endl;
    }
  }
};



#ifdef WIN32
template <class T>  
class SpatialHashMap : public hash_map< Cell, CellContents<T>, cell_hash_compare>{
#else
template <class T>  
class SpatialHashMap : public hash_map< Cell, CellContents<T>, hash_cell, equal_cell >{
#endif

 private:
  scalar cellsize;
  scalar cellsize_inverse;

  CellContents<T> empty_cell;

 public:
  SpatialHashMap(scalar csize) {
    assert(csize > 0);
    cellsize = csize;
    cellsize_inverse = (scalar) 1.0/ (scalar)csize;
  }

  const CellContents<T>& GetCellContents(const Cell& c) const{
    typename SpatialHashMap::const_iterator i = SpatialHashMap::find(c);
    if(i == SpatialHashMap::end())
      return empty_cell;
    else 
      return (*i).second;
  }

  const CellContents<T> * GetCellContents(const Vector3DMeshVersion& x) const{
    return GetCellContents(toCellInverse(x,cellsize_inverse));
  }

  //  Particle&  operator[](const Particle& p) const{ return array[i]; }

  scalar CellSize() const        { return cellsize; } 
  scalar CellSizeInverse() const { return cellsize_inverse; } 

  Cell toCell(const Vector3DMeshVersion& x) const { return toCellInverse(x,cellsize_inverse); }

  void insert(const Vector3DMeshVersion& vec, const T& t){ insert(toCell(vec),t);  }
  void remove(const Vector3DMeshVersion& vec, const T& t){ remove(toCell(vec),t);         }

  void insert(const Cell& c, const T& t){ (*this)[c].Add(t);  }
  void remove(const Cell& c, const T& t){ 
    typename SpatialHashMap::iterator i = SpatialHashMap::find(c);
    if(i != SpatialHashMap::end()){
      i->second.Remove(t);
      if(i->second.Empty()){
	SpatialHashMap::erase(i);
      }
    }
  }


  //  bool contains(Cell c, const RigidParticle * ptr); 
};




/*
 * VoxelNeighborhoodMap
 */


/* typedef void* VOIDP; */

/* struct hash_void{ */
/*   size_t operator()(const VOIDP & c) const{ */
/*     return (size_t)((int) c + 1013234 * (int) c); */
/*   } */
/* }; */

/* struct equal_void{ */
/*   bool operator()(const VOIDP& a, const VOIDP& b) const{ */
/*     return a == b; */
/*   } */
/* }; */


/* #ifdef WIN32 */
/* typedef hash_set<unsigned int> VoxelHashSet; */
/* #else */
/* typedef hash_set<void * ,hash_void,equal_void> VoxelHashSet; */
/* #endif */


/* template<class T>  */
/* class VoxelCache{ */
/*  public: */
/*   VoxelHashSet hset; */
/*   vector<T *> cache; */
/*   bool dirty; */

/*   VoxelCache(){ */
/*     dirty = false; */
/*   } */

/*   ~VoxelCache(){ */
/*     cache.clear(); */
/*     hset.clear(); */
/*   } */

/*   vector< T * > * getContents(){ */
/*     if(dirty){ */
/*       dirty = false; */
      
/*       unsigned int size = hset.size(); */
      
/*       cache.resize(size); */
      
/*       VoxelHashSet::const_iterator i   = hset.begin(); */
/*       for(unsigned int n = 0; n < size; n++){ */
/* 	cache[n] = (T *) (*i); */
/* 	i++; */
/*       } */
/*     } */
/*     //    cout << cache.size() << endl; */
/*     return &cache;     */
/*   } */
  

/*   bool empty() const{ */
/*     return hset.empty(); */
/*   }   */

/*   void add(const T * e){ */
/*     dirty = true; */
/*     hset.insert((void *)e);       */
/*   } */

/*   void remove(const T * e){ */
/*     dirty = true; */
/*     hset.erase((void *)e); */
/*   } */

    
/*   unsigned int size() const{ */
/*     return hset.size(); */
/*   } */

/*   bool contains(const T * e){ */
/*     if(hset.count((unsigned int)e) == 0){ */
/*       return false; */
/*     } else { */
/*       return true; */
/*     } */
/*   }  */
/* }; */




/* template <class T >  */
/* class VoxelNeighborhoodMap{ */
/*  private: */
/*   scalar cellsize; */
/*   scalar cellsize_inverse; */

/* #ifdef WIN32 */
/*   typedef  hash_map< Cell, VoxelCache<T>, cell_hash_compare> voxelcachemap; */
/* #else */
/*   typedef  hash_map< Cell, VoxelCache<T> , hash_cell, equal_cell >  voxelcachemap; */
/* #endif */
/*   voxelcachemap map; */

/*   vector< T *> emptyvector; */

/*  public: */
  
/*   VoxelNeighborhoodMap(scalar cell_size){ assert(cell_size > 0.0); cellsize = cell_size, cellsize_inverse = (scalar) 1.0 / cellsize; } */
  
/*   void insert(const Cell& c, const T * tp){ */
/*     for(int i = -1; i <= 1; i++){ */
/*       for(int j = -1; j <= 1; j++){ */
/* 	for(int k = -1; k <= 1; k++){ */
/* 	  insertSingle(c + Cell(i,j,k),tp); */
/* 	} */
/*       } */
/*     } */
/*   } */

/*   void erase(const Cell& c, const T * tp){ */
/*     for(int i = -1; i <= 1; i++){ */
/*       for(int j = -1; j <= 1; j++){ */
/* 	for(int k = -1; k <= 1; k++){ */
/* 	  eraseSingle(c + Cell(i,j,k),tp); */
/* 	} */
/*       } */
/*     } */
/*   } */

/*   vector< T * > * neighbors(const Cell& c){ */
/*     typename voxelcachemap::iterator i = map.find(c);  */
    
/*     if(i != map.end()){  */
/*       return i->second.getContents();  */
/*     } else {  */
/*       return &emptyvector;  */
/*     }  */
/*   }  */

/*   Cell toCell(const Vector3DMeshVersion &v) const { return toCellInverse(v,cellsize_inverse); } */

/*  protected: */
/*   void insertSingle(const Cell& c, const T * tp){ */
/*     map[c].add(tp); */
/*   } */

/*   void eraseSingle(const Cell& c, const T * tp){ */
/*     map[c].remove(tp); */
/*     if(map[c].size() == 0){ */
/*       map.erase(c); */
/*     } */
/*   } */
     
/* }; */


#endif
