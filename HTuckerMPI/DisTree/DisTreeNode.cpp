/*
 * Parallel Distrubuted Tree Node implementation by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "DisTreeNode.h"
#ifndef NULL
	#define NULL 0
#endif

/*
 * Locally used helper functions:
 */

/*
 * Static Local Variables:
 */

/*
 * Function Implementations: eg:
 * Object Class::foo(Object in1, Object& in2);
 */
DisTreeNode::DisTreeNode(){
	this->t_id = 0;
	this->p_id = 0;
	this->l_id = 0;
	this->r_id = 0;
}

DisTreeNode::DisTreeNode(	Tensor& D,
							std::valarray<lapack_int>& index,
							int taskID, int parID, int leftID, int rightID){
	this->D = D;
	this->index = index;
	this->t_id = taskID;
	this->p_id = parID;
	this->l_id = leftID;
	this->r_id = rightID;
}

DisTreeNode& DisTreeNode::operator=(const DisTreeNode& E){
	this->D = E.D;
	this->index = E.index;
	this->t_id = E.t_id;
	this->p_id = E.p_id;
	this->l_id = E.l_id;
	this->r_id = E.r_id;
	return *this;
}

void DisTreeNode::printID() const{
	printf("\nt_id:\t%d\np_id:\t%d\nl_id:\t%d\nr_id:\t%d\n",
			this->t_id,this->p_id,this->l_id,this->r_id);
}

int DisTreeNode::getP() const{
	return this->p_id;
} 

int DisTreeNode::getL() const{
	return this->l_id;
}
 
int DisTreeNode::getR() const{
	return this->r_id;
}
