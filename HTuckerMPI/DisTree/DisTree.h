#ifndef DISTREE_H
#define DISTREE_H

/*
 * C Header file Template by Bram Rodgers.
 * Original Draft Dated: 25, Feb 2018
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include"DisTreeNode.h"
#include<stdlib.h>
#ifndef NULL
	#define NULL 0
#endif

//This object is used to handle the nodes given above.
class DisTree{
	private:
	public:
	//The array which contains all the tree nodes
	std::valarray<DisTreeNode> tree;
	
	/*
	 * Fills balanced tree where the root index size is given
	 *
	 */
	DisTree(lapack_int rootIndexSize);
	
	/*
	/
	 * Returns a valarray<DisTreeNode*> object of all object in layer l, given
	 * Since the layers are stored contiguously, it just has to get pointers 
	 * to a a sub array of DisTree:tree
	 /
	std::valarray<DisTreeNode*> getLayer(lapack_int l);
	*/ 
	
	/*
	 *	Prints all the elements in the tree out onto the screen vertically
	 *	Prints the right children first, then the left children
	 */
	void printTree();
};
#endif
