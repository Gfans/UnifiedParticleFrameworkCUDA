#ifndef  GPU_UNIFIED_ZINDEX_H_
#define  GPU_UNIFIED_ZINDEX_H_

#include <vector>

#include <vmmlib/vector3.h>
#include "UnifiedConstants.h"

// we only need 10 bits to represent a particle's grid position along any direction in our demos
// all bits of z z-index in this case can be packed into a single 32-bit unsigned integer
// 32-bit z-indexing masks
const unsigned int XMASK = 0xB6DB6DB6;
const unsigned int YMASK = 0x6DB6DB6D;
const unsigned int ZMASK = 0xDB6DB6DB;

#define XINC(i) ( (((i | XMASK) + 1) & ~XMASK) | (i & XMASK) )
#define YINC(i) ( (((i | YMASK) + 1) & ~YMASK) | (i & YMASK) )
#define ZINC(i) ( (((i | ZMASK) + 1) & ~ZMASK) | (i & ZMASK) )

typedef vmml::Vector3<unsigned short int> Vector3ui;
typedef std::vector< std::pair<unsigned int,unsigned int> > ZRanges;


class ZIndex {
public:
  
  ZIndex(const int size=8);
  
  unsigned int *z_table()
  {
    return &z_table_[0][0];
  }
  
  unsigned int block_size() const
  {
    return block_size_;
  }
  
  // generate z-index for given 3D coordinate, only 16-bit integer coordinates supported
  inline unsigned int CalcIndex(const unsigned short int x, const unsigned short int y, const unsigned short int z)
    { return 0 | z_table_[x][0] | z_table_[y][1] | z_table_[z][2]; }
  inline unsigned int CalcIndex(const unsigned short int coordinate[3])
	{ return 0 | z_table_[coordinate[0]][0] | z_table_[coordinate[1]][1] | z_table_[coordinate[2]][2]; }
  inline unsigned int CalcIndex(const Vector3ui coordinate)
	{ return 0 | z_table_[coordinate[0]][0] | z_table_[coordinate[1]][1] | z_table_[coordinate[2]][2]; }
  
  // query for octree cell index-ranges contained in given box, block_size_ indicates power-of-two desired cell size
  void SetBlockSize(const int size);
  void BoxQuery(const Vector3ui min, const Vector3ui max, ZRanges &ranges);

private:
  unsigned int block_size_;
  unsigned int z_table_[1024][3];
};

#endif	// GPU_UNIFIED_ZINDEX_H_
