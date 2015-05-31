/** Author: J. Andreas Bærentzen, (c) 2003
		This is a simple KDTree data structure. It is written using templates, and
		it is parametrized by both key and value. 

   Modified and adapted by Nathan Bell (March 25 2005)
*/
 
#ifndef KDTREE_H
#define KDTREE_H

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

#include "scalar.h"
#include "bits.h"

/** A classic K-D tree. 
    A K-D tree is a good data structure for storing points in space
    and for nearest neighbour queries. It is basically a generalized 
    binary tree in K dimensions. */

typedef short disc_type;

template<class KeyT, class ValT>
class KDTree
{
	typedef KeyT KeyType;
	typedef std::vector<KeyT> KeyVectorType;
	typedef std::vector<ValT> ValVectorType;
		

	/// KDNode struct represents node in KD tree
	struct KDNode
	{
		KeyT key;
		ValT val;
	        disc_type dsc;

		KDNode(): dsc(0) {}

		KDNode(const KeyT& _key, const ValT& _val):
			key(_key), val(_val), dsc(-1) {}

		scalar dist(const KeyType& p) const 
		{
			KeyType dist_vec = p;
			dist_vec  -= key;
			return dist_vec.norm2();
		}
	};

	typedef std::vector<KDNode> NodeVecType;
	NodeVecType init_nodes;
	NodeVecType nodes;


	// PQNode struct represents node in priority queue
	struct PQNode
	{
	        KeyT key;
		ValT val;
	        scalar dist;

		PQNode(const KeyT& _key, const ValT& _val, const scalar& _dist):
			key(_key), val(_val), dist(_dist) {}
	};
	

	class PQNodeComp
	  {
	  public:
	    bool operator()(const PQNode& a, const PQNode& b) const
	      {
		return a.dist < b.dist;
	      }
	  };

	typedef std::priority_queue<PQNode, std::vector<PQNode> ,PQNodeComp> NodeQueue;


	/// The greatest depth of KD tree.
	unsigned int max_depth;

	/// Total number of elements in tree
	unsigned int elements;

	/** Comp is a class used for comparing two keys. Comp is constructed
			with the discriminator - i.e. the coordinate of the key that is used
			for comparing keys - Comp objects are passed to the sort algorithm.*/
	class Comp
	{
		const disc_type dsc;
	public:
		Comp(disc_type _dsc): dsc(_dsc) {}
		bool operator()(const KeyType& k0, const KeyType& k1) const
		{
			unsigned int dim=KeyType::Dimension();
			for(unsigned int i=0;i<dim;i++)
				{
					int j=(dsc+i)%dim;
					if(k0[j]<k1[j])
						return true;
					if(k0[j]>k1[j])
						return false;
				}
			return false;
		}

		bool operator()(const KDNode& k0, const KDNode& k1) const
		{
			return (*this)(k0.key,k1.key);
		}
	};

	/// The dimension -- K
	const unsigned int DIM;

	/** Passed a vector of keys, this function will construct an optimal tree.
			It is called recursively - second argument is level in tree. */
	void optimize(unsigned int, unsigned int, unsigned int, unsigned int);

	/** Find nearest neighbour. */
	unsigned int closest_point_priv(unsigned int, const KeyType&, scalar&) const;
	
	void nearest_n_priv(unsigned int,const KeyType&, unsigned int, scalar&, NodeQueue&) const;
	void nearest_n_priv_helper(const KeyType&, unsigned int, scalar&, KeyVectorType&, ValVectorType&) const;


	/** Finds the optimal discriminator. There are more ways, but this 
			function traverses the vector and finds out what dimension has
			the greatest difference between min and max element. That dimension
			is used for discriminator */
	disc_type opt_disc(unsigned int,unsigned int) const;


public:

	/** Build tree from vector of keys passed as argument. */
	KDTree():
	  max_depth(0), elements(0), DIM(KeyType::Dimension())
	{
	}

	void insert(const KeyT& key, const ValT& val)
	{
		init_nodes.push_back(KDNode(key,val));
	}

	void build()
	{
		nodes.resize(init_nodes.size()+1);
		if(init_nodes.size() > 0)	
			optimize(1,0,init_nodes.size(),0);
 		NodeVecType v(0);
 		init_nodes.swap(v);
	}

 	bool closest_point(const KeyT& p, KeyT&k, ValT&v) const{
	  if(nodes.empty()){ return false; }

	  scalar max_sq_dist = SCALAR_MAX;
		if(unsigned int n = closest_point_priv(1, p, max_sq_dist))
			{
				k = nodes[n].key;
				v = nodes[n].val;
				return true;
			}
		return false;

	}

 	bool closest_point(const KeyT& p, scalar& dist, KeyT&k, ValT&v) const
	{
	  if(nodes.empty()){ return false; }

	  scalar max_sq_dist = dist * dist;
		if(unsigned int n = closest_point_priv(1, p, max_sq_dist))
			{
				k = nodes[n].key;
				v = nodes[n].val;
				dist = SQRT(max_sq_dist);
				return true;
			}
		return false;
	}

	void in_sphere(unsigned int n, 
		       const KeyType& p, 
		       const scalar& dist,
		       KeyVectorType& keys,
		       ValVectorType& vals) const;
	
	int in_sphere(const KeyType& p, 
		      scalar dist,
		      KeyVectorType& keys,
		      ValVectorType& vals) const
	{
		scalar max_sq_dist = dist*dist;
		in_sphere(1,p,max_sq_dist,keys,vals);
		return keys.size();
	}
	
	int nearest_n(const KeyType& p,
		      unsigned int max_N,
		      KeyVectorType& keys,
		      ValVectorType& vals) const
	  {
	    scalar max_sq_dist = SCALAR_MAX;
	    nearest_n_priv_helper(p,max_N,max_sq_dist,keys,vals);
	    return keys.size();
	  }
	
	int nearest_n(const KeyType& p,
		      unsigned int max_N,
		      scalar max_dist,
		      KeyVectorType& keys,
		      ValVectorType& vals) const
	  {
	    scalar max_sq_dist = max_dist * max_dist;
	    nearest_n_priv_helper(p,max_N,max_sq_dist,keys,vals);
	    return keys.size();
	  }
	

};

template<class KeyT, class ValT>
disc_type KDTree<KeyT,ValT>::opt_disc(unsigned int kvec_beg,  
				unsigned int kvec_end) const 
{
	KeyType vmin = init_nodes[kvec_beg].key;
	KeyType vmax = init_nodes[kvec_beg].key;
	for(unsigned int i=kvec_beg;i<kvec_end;i++)
		{
		  for(unsigned int j=0; j < KeyType::Dimension(); j++){
		    if(init_nodes[i].key[j] < vmin[j])
		      vmin[j] = init_nodes[i].key[j];
		    else if(init_nodes[i].key[j] > vmax[j])
		      vmax[j] = init_nodes[i].key[j];
		  }
		}
	int od=0;
	KeyType ave_v = vmax-vmin;
	for(unsigned int i=1;i<KeyType::Dimension();i++)
		if(ave_v[i]>ave_v[od]) od = i;
	return od;
} 


template<class KeyT, class ValT>
void KDTree<KeyT,ValT>::optimize(unsigned int cur, unsigned int kvec_beg, unsigned int kvec_end, unsigned int level)
{
	// Assert that we are not inserting beyond capacity.
	assert(cur < nodes.size());


	int N = kvec_end-kvec_beg;

	// If there is just a single element, we simply insert.
	if(N == 1) {
	  max_depth  = (level > max_depth) ? level : max_depth;
	  nodes[cur] = init_nodes[kvec_beg];
	  nodes[cur].dsc = -1;
	  return;
	}
	
	// Find the axis that best separates the data.
	disc_type disc = opt_disc(kvec_beg, kvec_end);

	// Compute the median element. See my document on how to do this
	// www.imm.dtu.dk/~jab/publications.html

	unsigned int M = 1 << (integer_log2(N));
	unsigned int R = N-(M-1);
	unsigned int left_size  = (M-2)/2;
	unsigned int right_size = (M-2)/2;
	
	if(R < M/2){
	  left_size += R;
	}
	else {
	  left_size += M/2;
	  right_size += R-M/2;
	}
	unsigned int median = kvec_beg + left_size;
	

	// Sort elements but use nth_element (which is cheaper) than
	// a sorting algorithm. All elements to the left of the median
	// will be smaller than or equal the median. All elements to the right
	// will be greater than or equal to the median.
	const Comp comp(disc);
	std::nth_element(&init_nodes[kvec_beg], &init_nodes[median], &init_nodes[kvec_end], comp);

	// Insert the node in the final data structure.
	nodes[cur] = init_nodes[median];
	nodes[cur].dsc = disc;

	// Recursively build left and right tree.
	//	if(left_size>0)	// <- left size must be > 0
	optimize(2*cur, kvec_beg, median,level+1);

	if(N > 2) 
	  optimize(2*cur+1, median+1, kvec_end,level+1);
}




template<class KeyT, class ValT>
void KDTree<KeyT,ValT>::nearest_n_priv_helper(const KeyType& p, unsigned int max_N, scalar& dist, 
					      KeyVectorType& keys, ValVectorType& vals) const{
  NodeQueue queue;
  nearest_n_priv(1,p,max_N,dist,queue);
  
  keys.resize(queue.size());
  vals.resize(queue.size());

  unsigned int i = queue.size() - 1;

  while(!queue.empty()){
    const PQNode& top = queue.top();
    keys[i] = top.key;
    vals[i] = top.val;

    queue.pop();
    i--; 
  }
}

template<class KeyT, class ValT>
void KDTree<KeyT,ValT>::nearest_n_priv(unsigned int n, const KeyType& p, unsigned int max_N, scalar& dist, NodeQueue& queue) const{
  const KDNode& node = nodes[n];

  scalar this_dist = node.dist(p);

  if(this_dist < dist){
    if(queue.size() == max_N){
      queue.pop();
    }

    queue.push(PQNode(node.key,node.val,this_dist));

    if(queue.size() == max_N){
      dist = queue.top().dist;
    }
  }

  if(node.dsc != -1){ //there are children
    int dsc = node.dsc;
    scalar dsc_dist  = (node.key[dsc]-p[dsc]); dsc_dist *= dsc_dist;
    bool left_son = Comp(dsc)(p,node.key);
    
    if(left_son || dsc_dist < dist){
      const unsigned int left_child = 2*n;

      //Since this is not a leaf we must have a left child, no check needed
      nearest_n_priv(left_child, p, max_N, dist, queue); 	
    }

    if(!left_son || dsc_dist < dist){    
      const unsigned int right_child = 2*n+1;

      //However we must always check for a right child
      if(right_child < nodes.size())                           
	nearest_n_priv(right_child, p, max_N, dist, queue);
    }
  }

}

template<class KeyT, class ValT>
unsigned int KDTree<KeyT,ValT>::closest_point_priv(unsigned int n, const KeyType& p, scalar& dist) const {
	unsigned int ret_node = 0;
         scalar this_dist = nodes[n].dist(p);
	 

	if(this_dist<dist)
		{
			dist = this_dist;
			ret_node = n;
		}
	if(nodes[n].dsc != -1)
		{
			int dsc         = nodes[n].dsc;
			scalar dsc_dist  = (nodes[n].key[dsc]-p[dsc]); dsc_dist *= dsc_dist;
			bool left_son   = Comp(dsc)(p,nodes[n].key);

			if(left_son||dsc_dist<dist)
				{
					const unsigned int left_child = 2*n;

					//Since this is not a leaf we must have a left child, no check needed
					if(unsigned int nl=closest_point_priv(left_child, p, dist))
					  ret_node = nl;
				}
			if(!left_son||dsc_dist<dist)
				{
					const unsigned int right_child = 2*n+1;

					//However we must always check for a right child
					if(right_child < nodes.size())
						if(unsigned int nr=closest_point_priv(right_child, p, dist))
							ret_node = nr;
				}
		}
	return ret_node;
}

template<class KeyT, class ValT>
void KDTree<KeyT,ValT>::in_sphere(unsigned int n, 
				  const KeyType& p, 
				  const scalar& dist,
				  std::vector<KeyT>& keys,
				  std::vector<ValT>& vals) const
{
	scalar this_dist = nodes[n].dist(p);
	assert(n<nodes.size());
	if(this_dist<dist)
		{
			keys.push_back(nodes[n].key);
			vals.push_back(nodes[n].val);
		}
	if(nodes[n].dsc != -1)
		{
			const int dsc         = nodes[n].dsc;
			scalar dsc_dist  = (nodes[n].key[dsc]-p[dsc]); dsc_dist *= dsc_dist;

			bool left_son = Comp(dsc)(p,nodes[n].key);

			if(left_son||dsc_dist<dist)
				{
					unsigned int left_child = 2*n;
					if(left_child < nodes.size())
						in_sphere(left_child, p, dist, keys, vals);
				}
			if(!left_son||dsc_dist<dist)
				{
					unsigned int right_child = 2*n+1;
					if(right_child < nodes.size())
						in_sphere(right_child, p, dist, keys, vals);
				}
		}
}



#endif
