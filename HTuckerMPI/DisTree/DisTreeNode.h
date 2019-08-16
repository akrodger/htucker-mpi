#ifndef DISTREENODE_H
#define DISTREENODE_H

/*
 * Parallel Distrubuted Tree Structure header by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include<stdio.h>
#include"../Tensor/Matrix/Matrix.h"
#ifndef NULL
	#define NULL 0
#endif

/*
 * Object and Struct Definitions:
 */

/* This is a distributed memory binary tree parallel data structure implemented
 * using OpenMPI. The data structure has 6 pieces of data.
 *
 * ---4 integers: (All Public)---
 * One of the task ID of this node,
 * one for task ID of the parent,
 * one for the task ID of the left child, and one
 * for the task ID of the right child. We use the folllowing ID conventions:
 * 	
 *	l_id = (task_id)*2 + 1
 *	r_id = (task_id)*2 + 2
 *	If task_id  is odd:
 *		p_id = (task_id - 1) / 2
 * 	else:
 *		p_id = task_id / 2
 *
 * To deal with the last layer, use the [0] and [1] indexes of the second to
 * last layer. Essentially: Let p be the number of layers in the tree.
 * 							Let l_id be as defined above.
 *	if(l_id > 2^(p-1) - 2 && this->index.size() == 2)
 *		//if l_id is in last layer, and thus so is r_id, and this node splits
 *		l_id = 2^(p-1) - 1 + this->index[0]
 *		r_id = 2^(p-1) - 1 + this->index[1]
 *		//then use the index values to cram in the tree to the end of the array
 *
 * Note that if p_id == task_id, then task_id = 0. i.e. this task is the root
 *
 * ---An instance of an object of type Tensor: (Public)---
 * This object, called D, is the data stored at one node on the tree. It will
 * be used to store transfer tensors and basis matrices.
 *
 * ---A instance of an object of type valarray<lapack_int>: (Public)---
 * This object stores a collection of integers which correspond to which
 * matricization the current node corresponds to. Example: the root corresponds
 * to the vectorization of a tensor. The leaves correspond to the singleton
 * matricizations.
 */
class DisTreeNode{
	private:

	public:
	//Parent, Left, and Right ID numbers
	int t_id, p_id, l_id, r_id;
	//Data Member:
	Tensor D;
	//index on the binary tree
	std::valarray<lapack_int> index;
	
	DisTreeNode();

	DisTreeNode(	Tensor& D,
					std::valarray<lapack_int>& index,
					int taskID, int parID, int leftID, int rightID);
	
	//equality operator for nodes
	DisTreeNode& operator=(const DisTreeNode& E);

	//Print out the ID data associated with this node
	void printID() const;

	//Get index of parent
	int getP() const;
	
	//Get index of left child
	int getL() const;
	
	//Get index of right child
	int getR() const;
};


/*
 * Function Declarations:
 */


#endif
