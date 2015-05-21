#include <stdio.h>
#include <fcntl.h>
#include <assert.h>

#include <vmmlib/vector3.h>

#include "zIndex.h"


ZIndex::ZIndex(const int size)
{
  int i, k;
  
  block_size_ = size;
  
  for (k = 0; k < 1024; k++) {
	z_table_[k][0] = z_table_[k][1] = z_table_[k][2] = 0;
	for (i = 0; i < 10; i++) {
	  z_table_[k][0] = z_table_[k][0] | ((((unsigned int)k >> i) & 0x00000001) << (3*i));
	  z_table_[k][1] = z_table_[k][1] | ((((unsigned int)k >> i) & 0x00000001) << (3*i+1));
	  z_table_[k][2] = z_table_[k][2] | ((((unsigned int)k >> i) & 0x00000001) << (3*i+2));
	}
  }
}


void ZIndex::SetBlockSize(const int size)
{
  // block size should be a power of two
  for (block_size_ = 1; block_size_ < size; block_size_ *= 2) 
	  ;

#ifdef SPH_DEMO_SCENE_2
  /*int limitSize = 16;
  if( block_size_ < limitSize) 
	block_size_ = limitSize;*/
#endif

   std::cout << "block size : " << block_size_  << std::endl;
}


void ZIndex::BoxQuery(const Vector3ui min, const Vector3ui max, ZRanges &ranges)
{
	const unsigned int offset = block_size_*block_size_*block_size_;		// S * S * S
	Vector3ui start(min);
	Vector3ui end(max);

	// get cell corners of query box for given block size
	start /= Vector3ui(block_size_);
	end /= Vector3ui(block_size_);

	// set size for result array
	vmml::Vector3i blocks(1);
	blocks += (end - start);
	ranges.resize(blocks.x * blocks.y * blocks.z);

	// get index ranges of cells covered by given min_query_box & max_query_box
	int cnt = 0;
	unsigned int xind = CalcIndex(start);
	for (int i = start.x; i <= end.x; i++) {
		unsigned int yind = xind;
		for (int j = start.y; j <= end.y; j++) {
			unsigned int zind = yind;
			for (int k = start.z; k <= end.z; k++) {
				// The starting Z-index s of any block of size S can easily be determined and particles falling into that block form a sequence between s and s+S3
				ranges[cnt++] = std::pair<unsigned int,unsigned int>(offset * zind, offset * (zind + 1) - 1);   // s + S * S * S
				zind = ZINC(zind);
			}
			yind = YINC(yind);
		}
		xind = XINC(xind);
	}
}
