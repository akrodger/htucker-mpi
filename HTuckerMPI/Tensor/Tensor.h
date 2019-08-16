#ifndef TENSOR_H
#define TENSOR_H

/*
 * Dense Tensor Object header by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
//Define the lapack_int label to turn on long ints
#ifndef lapack_int
	#define lapack_int long
#endif
#ifndef lapack_logical
	#define lapack_logical int
#endif
#include<valarray>
#include<stdlib.h>
#include"stdio.h"
#include"string.h"
#include"cblas.h"
#include"lapacke.h"
#include"math.h"
#include"omp.h"
#ifndef NULL
	#define NULL 0
#endif
#ifndef INTEL_WORD_SIZE
	#define INTEL_WORD_SIZE 64
#endif

#ifndef EPS_M
	#define EPS_M 1.11e-16
#endif

#ifndef MIN
	#define MIN(X,Y) ((X) <= (Y) ? (X) : (Y))
#endif

#ifndef MAX
	#define MAX(X,Y) ((X) >= (Y) ? (X) : (Y))
#endif

#ifndef ABS
	#define ABS(X) ((X) < 0? -(X) : (X))
#endif

#ifndef SIGN
	#define SIGN(X) ((X)/ABS(X))
#endif

//namespace moctalab{
/*
 * Object and Struct Definitions:
 *  COLUMN MAJOR FORM TENSOR!!!
 */
class Tensor{
private:
	std::valarray<lapack_int> indexLength;
	lapack_int dim;
	lapack_int numComponents;
public:
	/* This the pointer to address of the components of this Tensor
	 * This is used for LAPACK access. It is not recommended for use
	 * outside of this access to LAPACK. You have been warned.
	 */
	std::valarray<double> components;
/*
 * Function Declarations:
 */
	//Constructors:
	Tensor();
	/*
	 * sizeVector is array of max indexes. dimension is the number of elements
	 * in sizeVector.
	 */
	Tensor(std::valarray<lapack_int>& sizeVector, lapack_int dimension);

	/*
	 *	Tensor Copy constructor. Initializes a new Tensor with all of the
	 *	same data as the old tensor.
	 */
	Tensor(const Tensor& copyFrom);

	//Destructors:
	~Tensor();
	/*
	 * Initialize a Tensor, filled with garbage entries:
	 */
	lapack_int init(const std::valarray<lapack_int>& sizeVector,
					lapack_int dimension);

	/*
	 * Get value at multiIndex. Throw an error if out of bounds 
	 * or if multiIndex == NULL. Tensor::operator() is the 
	 * same as Tensor::entry()
	 */
	double& entry(std::valarray<lapack_int>& multiIndex);

	double& operator()(std::valarray<lapack_int>& multiIndex);

	//Assignment/Copy Operator
	Tensor& operator=(const Tensor& B);

	/*
	 * Scalar Multiplication of a Tensor by a scalar double called a.
	 */
	Tensor operator*(const double a) const;

	/*
	 * Scalar division of a Tensor by a scalar double called a.
	 */
	Tensor operator/(const double a) const;

	/*
 	 * Addition of Two Tensors component-wise.
	 */
	Tensor operator+(const Tensor& B) const ;

	/*
	 * Addition of tensor and a scalar
	 */
	Tensor operator+(const double a) const;

	Tensor operator-(const Tensor& B) const;

	Tensor operator-(const double a) const;
	/*
	 * Const compatible access to an element of the Tensor
	 */
	double get(std::valarray<lapack_int>& multiIndex) const;

	//pass by value the [k] entry of the components arrow for this matrix
	double get(lapack_int k) const;

	/*
	 *	Create a new tensor which has *this and B on the block diagonal.
	 *	Matlab Example:
	 *	Let A and B be matrices. A.diagCat(B) = [A, 0 ;0, B]
	 */
	Tensor diagCat(const Tensor& B) const;

	/*
	 *	The the length of the i^th multi-index
	 */
	lapack_int getIndexLength(lapack_int i) const;

	/*
	 * The number of components in the tensor
	 */
	lapack_int getNumComponents() const;
	/*
	 * Get the dimension of the Tensor:
	 */
	lapack_int getDim() const;
	/*
	 * Count a multiIndex up by 1 value in its alphabet.
	 */
	//This version says that we want to iterate on a subset of the give indexes
	void multiIterate(	std::valarray<lapack_int>& iter,
						const std::valarray<lapack_int>& indexBound,
						const std::valarray<lapack_int>& iterSubset,
						lapack_int subsetSize) const;
	//This version says that we want to iterate over all indexes
	//Pass NULL into the 3rd argument
	void multiIterate(	std::valarray<lapack_int>& iter,
						const std::valarray<lapack_int>& indexBound,
						void*,
						lapack_int iterSize) const;
};

//};//end namespace moctalab
#endif
