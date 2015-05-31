#ifndef ALGO_H
#define ALGO_H


#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

template <class PairType> class __CompPair
{
 public:
  bool operator()(const PairType& a, const PairType& b) const
    {
      return a.first < b.first;
    }
};


template <typename ValueIterator, typename IndexIterator>
  void
  nth_element_pair(ValueIterator __first,
		   ValueIterator __nth,
		   ValueIterator __last,
		   IndexIterator __start)
{

  typedef typename iterator_traits<ValueIterator>::value_type ValueType;
  typedef typename iterator_traits<IndexIterator>::value_type IndexType;

  typedef pair<ValueType, IndexType> ValueIndexPair;
  typedef vector<ValueIndexPair> PairVector;

  PairVector pairs;

  ValueIterator i;
  IndexIterator j;

  for(i = __first, j = __start; i != __last; i++, j++){
    pairs.push_back(ValueIndexPair(*i,*j));
  }


  __CompPair<ValueIndexPair> Comp;
  nth_element(pairs.begin(), pairs.begin() + (__nth - __first), pairs.end(), Comp);


  unsigned int k;
  for(i = __first, j = __start, k = 0; i != __last; i++, j++, k++){
    *i = pairs[k].first;
    *j = pairs[k].second;
  }
  
}

#endif
