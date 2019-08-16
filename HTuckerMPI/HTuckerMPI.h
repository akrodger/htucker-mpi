#ifndef HTUCKERMPI_H
#define HTUCKERMPI_H

/*
 * HTucker MPI Parallel header by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */
/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include"mpi.h"
#include"DisTree/DisTree.h"
#ifndef NULL
	#define NULL 0
#endif

/*
 * Object and Struct Definitions:
 */
class HTuckerMPI{
private:
public:
	//the indexLength object tells you what the dimensions of the full
	//tensor corresponding to this HTucker object is
	std::valarray<lapack_int> indexLength;
	DisTreeNode n;
	HTuckerMPI();
	//Constructor which makes an HTuckerMPI object with no data stored
	//Just a tree with the given tensor dimension. 
	HTuckerMPI(std::valarray<lapack_int>& sizeVector, lapack_int tensorDim);
	//Root to leaves constructor for HTucker Tensors.
	//Due to distributed memory processing, root to leaves is faster than
	//leaves to root.
	HTuckerMPI(Tensor& T, lapack_int maxRank);
	//assignment operator, copy argument to this object
	HTuckerMPI& operator=(const HTuckerMPI& E);
	//Addition operator. concatenates this object and argument B
	HTuckerMPI operator+(const HTuckerMPI& B) const;
	//multiply a copy of B by -1, then sum to this tensor.
	HTuckerMPI operator-(const HTuckerMPI& B) const;
	//Convert this HTucker tensor into a full object
	Tensor full();
	//Get the square of the 2-norm of this HTucker tensor. The 2-norm is
	//is the square root of the sum of all entries of the full tensor.
	//The tensor is orthogonalized in the process to given numerical stability.
	//The value is only returned on the root node, all other compute nodes
	//recieve the value zero.
	double norm2() const;
	//Get the inner product of 2 HTucker tensors. Uses similar algorithm
	//to the norm2() function.
	//the value is only returned on the root node, all other compute nodes
	//recieve the value zero.
	double innerProd(HTuckerMPI& x);
	//Do element-wise multiplication of full tensors, aka Hadamard product.
	//all in HTucker format
	HTuckerMPI hada(HTuckerMPI& x);
	//Orthogonalize this object using QR factorization and message passing
	void orthog();
	//Return a matrix containing the Gramian corresponding to this compute
	//node. Uses message passing to achieve this
	Matrix gramians();
	//Call orthog() and gramians(), then use the gramians to truncate this
	//HTuckerMPI tensor. Sets the ranks on the tree equal to maxRank
	void truncateHT(lapack_int maxRank);
	//Scale this node's transfer tensor by a double. i.e. multiply by that value
	void scale(double a);
	//Multiply a given matrix into this node's matrix. This is the HTucker
	//Form of mu-mode multiplication. If this is not a leaf, do nothing
	void leafMult(Matrix& L); 
	//This algorithm considers this instance to be an HTucker truncation
	//of a linear operator on the ambient tensor space in which the
	//argument x lives. It checks to see if the number of rows of the leaves
	//of x divide the number of rows of the leaves of this object.
	//If so, matrices are generated from the leaves and then applies to x.
	//on all other nodes, multiplication is carried out via a kronecker product
	HTuckerMPI linOpApply(HTuckerMPI& x);
	//Sums all the entries of an HTucker tensor together, except index k
	//which satisfies 0<=k && k<=this->indexLength.size() . This operation
	//is analogous to computing the maginal probability density of a
	//vector-valued random variable.
	Matrix contractedVector(lapack_int k);
	//Get the thread ID of this node:
	int getThreadID() const;
	//Get the i^th index value of this node's tree index
	lapack_int getTreeIndex(int i) const;
	//Get the index size of this node. Root will be equal to dimension.
	//leaves with have index length of 1.
	int getTreeIndexSize() const;
	//Print the data stored in DisTreeNode
	void print() const;
};

#endif
