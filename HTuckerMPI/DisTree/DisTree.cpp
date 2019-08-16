/*
 * Parallel Distrubuted Tree Structure implementation by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "DisTree.h"
#include<vector>
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

DisTree::DisTree(lapack_int rootIndexSize){
	//Find the number of layers, p
	long int pow_p = 1;
	int p = 0, 
		i = 0,
		j = 0,
		startOfLastLayer, 
		leftChild, 
		rightChild, 
		parent, 
		offSet,
		numNodes = 0;
	Tensor placeHolder = Tensor();
	//The below valarray<lapack_int>  is used to declare objects
	std::valarray<lapack_int> dec_index;
	std::vector<DisTreeNode> last_layer;
	DisTreeNode tempNode;
	//Especially simple case for a tree with 1 parent and 2 children:
	if(rootIndexSize == 2){
		dec_index.resize(rootIndexSize);
		dec_index[0] = 0;
		dec_index[1] = 1;
		this->tree.resize(3);
		this->tree[0] = DisTreeNode(placeHolder, dec_index, 0, 0, 1, 2);
		dec_index.resize(1);
		dec_index[0] = 0;
		this->tree[1] = DisTreeNode(placeHolder, dec_index, 1, 0, -1, -1);
		dec_index[0] = 1;
		this->tree[2] = DisTreeNode(placeHolder, dec_index, 2, 0, -1, -1);
		return;
	}
	while(pow_p < rootIndexSize){
		p++;
		numNodes += pow_p;
		pow_p *= 2;
	}
	numNodes += 2*rootIndexSize - pow_p;
	//Now we set the start index of last layer
	startOfLastLayer =  numNodes - 2*rootIndexSize + pow_p ;
	//The number of nodes in the tree is 2^(p-1) - 1 + rootIndexSize
	//since pow_p = 2^p, we have nodes = pow_p/2 -1 + rootIndexSize
	this->tree.resize((lapack_int)(numNodes));
	//now we set the index array of the root node
	dec_index.resize(rootIndexSize);
	//Now populate the root index with the values 0,1,...,rootIndexSize-1
	for(i = 0; i < rootIndexSize; i++){
		dec_index[i] = i;
	}
	//set the parent id and child id numbers of the root
	this->tree[0] = DisTreeNode(placeHolder, dec_index, 0, 0, 1, 2);
	//Now we can use the index i in the following loop to determine the index
	//arrays of every element in the tree
	//We iterate over all but the last layer of the tree.
	//In the second to last layer, we also make the last layer nodes
//	printf("\nnum inner nodes %ld\n", 
//		(lapack_int)(this->tree.size() - 2*rootIndexSize + pow_p) - 1);
	for(i = 1; i < (lapack_int)(startOfLastLayer); i++){
		//set proposed left and right child indexes, and parent
		leftChild = i*2 + 1;
		rightChild = i*2 + 2;
		parent = (i-1)/2;
		if(i % 2){//if this is a left child
			//resize declaration index to half of parent, rounded up
			dec_index.resize(this->tree[parent].index.size() - 
							(this->tree[parent].index.size()/2));
			offSet = 0;
		}else{//this is a right child
			//resize declaration index to half of parent, rounded down
			dec_index.resize((this->tree[parent].index.size()/2));
			offSet = (int)(this->tree[parent].index.size() - 
							(this->tree[parent].index.size()/2));
		}
		for(j = 0; j < (int)(dec_index.size()); j++){
			//set declaration index equal to left half of array
			dec_index[j] = this->tree[parent].index[j + offSet];
		}
		if(leftChild >= startOfLastLayer){//children would be in last layer
			//if this is a singleton node
			if(dec_index.size() == 1){
				this->tree[i] = DisTreeNode(placeHolder,
										dec_index, i, (i-1)/2, -1, -1);
			}else{//otherwise it has two singleton children
				this->tree[i] = DisTreeNode(placeHolder,
										dec_index, i, (i-1)/2,leftChild,
													 rightChild);
				//now declare its children
				//for this special case, set left and right child nodes
				//differently
				//load the children onto a stack
				leftChild = dec_index[0];
				rightChild = leftChild + 1;
				dec_index.resize(1);
				dec_index[0] = leftChild;
				tempNode = DisTreeNode(placeHolder,
										dec_index, leftChild, i, -1, -1);
				tempNode.p_id = i;
				last_layer.push_back(tempNode);
				dec_index[0] = rightChild;
				tempNode = DisTreeNode(placeHolder,
										dec_index, rightChild, i, -1, -1);
				tempNode.p_id = i;
				last_layer.push_back(tempNode);
			}
		}else{//This is a node on the interior of the tree
			this->tree[i] = DisTreeNode(placeHolder,
							dec_index, i, (i-1)/2, leftChild, rightChild);
		}
	}
	//Now load the stack of the last layer onto the end of the tree array
	for(i = startOfLastLayer; i < (int)(this->tree.size()); i++){
		this->tree[i] = last_layer[i - startOfLastLayer];
		if(i%2){
			this->tree[this->tree[i].p_id].l_id = i;
		}else{
			this->tree[this->tree[i].p_id].r_id = i;
		}
	}
}

void DisTree::printTree(){
	int currNode, poppedNode, j;
	std::vector<int> treeStack;
	//print each element in tree with recursive pattern
	currNode = 0;
	poppedNode = 0;
	treeStack.clear();
	/*
	for(i = 0;  i < (int)this->tree.size(); i++){
		printf("\naddr = %d,\t p_id = %d,\t"
				" l_id = %d,\t r_id = %d\nindex = \n",
				i, this->tree[i].p_id, 
					this->tree[i].l_id, this->tree[i].r_id);
		for(j = 0; j < (int)this->tree[i].index.size(); j++){
			printf("\t%d\n", this->tree[i].index[j]);
		}
	}*/
	while(1){
		while(currNode != -1){
			treeStack.push_back(currNode);
			currNode = this->tree[currNode].l_id;
		}
		if(currNode == -1 && treeStack.size() > 0){
			poppedNode = treeStack.back();
			treeStack.pop_back();
			currNode = this->tree[poppedNode].r_id;
			printf("\n");
			for(j = 0; j < (int)this->tree[poppedNode].index.size(); j++){
				printf("%ld", this->tree[poppedNode].index[j]);
			}
			printf("\n");
		}
		if(treeStack.size() == 0 && currNode == -1){
			break;
		}
	}
}

 
