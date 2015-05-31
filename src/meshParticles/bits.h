#ifndef BITS_H
#define BITS_H

/*
 * Useful bit twiddling hacks from http://graphics.stanford.edu/~seander/bithacks.html
 */

inline unsigned int integer_log2(unsigned int v){
  const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  const unsigned int S[] = {1, 2, 4, 8, 16};
  
  unsigned int c = 0; // result of log2(v) will go here
  for (int i = 4; i >= 0; i--) // unroll for speed...
    {
      if (v & b[i])
	{
	  v >>= S[i];
	  c |= S[i];
	} 
    }
  return c;
}


#endif
