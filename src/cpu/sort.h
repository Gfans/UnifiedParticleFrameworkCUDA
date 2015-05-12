#ifndef GPU_UNIFIED_H_
#define GPU_UNIFIED_H_

#include <cstdlib>
#include <algorithm>
#include <vector>

template< typename Item, typename Iterator, typename Compare >
class PivotMedianOfThree
{
public:
	inline Iterator operator()( Iterator left, Iterator right ) const
	{
		Iterator middle = left + ( ( right - left ) / 2 );
		if ( Compare()( *middle, *left ) )
			std::swap( *middle, *left );
		if ( Compare()( *right, *left ) )
			std::swap( *right, *left );
		if ( Compare()( *right, *middle ) )
			std::swap( *right, *middle );
		std::swap( *middle, *(right - 1 ) );
		return right - 1;
	}
};

template< typename Item, typename Iterator, typename Compare, typename FindPivot >
inline Iterator
	Partition( Iterator left, Iterator right )
{
	Iterator i    = left;
	Iterator j    = right;
	Item v        = *FindPivot()( left, right );

	while( true )
	{
		while( Compare()( *i, v ) )
		{
			++i;
		}
		while( Compare()( v, *j ) )
		{
			--j;
		}
		if ( i >= j )
			break;
		std::swap( *i, *j );
		--j;
		++i;
	}
	return j;
}

template< typename Item, typename Iterator, typename Compare, typename FindPivot >
inline void
	QuickSortInclusive( Iterator left, Iterator right )
{
	if ( right  <= left )
		return;

	Iterator i = Partition< Item, Iterator, Compare, FindPivot >( left, right );
	QuickSortInclusive< Item, Iterator, Compare, FindPivot >( left, i );
	QuickSortInclusive< Item, Iterator, Compare, FindPivot >( i + 1, right );
}

template< typename Item, typename Iterator, typename Compare, typename FindPivot >
inline void
	QuickSortTemplate( Iterator left, Iterator right )
{
	QuickSortInclusive< Item, Iterator, Compare, FindPivot >( left, right - 1 );
}

#endif	// GPU_UNIFIED_H_
