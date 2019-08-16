#ifndef MATRIX_H
#define MATRIX_H

/*
 * Dense Matrix Object header by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include"../Tensor.h"

//namespace moctalab{
/*
 * Object and Struct Definitions:
 */
class Matrix : public Tensor{
private:
public:
	//Empty Constructor:
	Matrix();

	//Basic Constructor
	Matrix(lapack_int numRows, lapack_int numCols);


	/* Common Types of Matrix Constructor.
	 * Set type = '0' for a matrix of zeros
	 * set type = 'I' for an idenetity matrix.
	 * set type = '1' for a matrix of all ones.
	 * set type = 'U' for uniform random entries in interval [0,1]
	 */
	Matrix(lapack_int numRows, lapack_int numCols, char initType);

	//Copy Constructor
	Matrix(const Matrix& copyFrom);

	/*
	 * Matricization Constructor
	 * Matricize along the *rowIndex indexes and store it in the Matrix object.
	 * rowIndexSize is the number of elements in *rowIndex.
	 */
	Matrix(const Tensor& T, std::valarray<lapack_int>& rowIndex, 
			lapack_int rowIndexSize);
	//Basic Destructor
	~Matrix();

	//pass by reference  the [row, col] entry of the matrix
	double& entry(lapack_int row, lapack_int col); 

	//pass by reference the [row, col] entry of the matrix
	double& operator()(lapack_int row, lapack_int col);

	//pass by reference the k^th component of the tensor. Used for
	//treating the tensor as a vector or array.
	double& operator()(lapack_int index);

	//pass by value the [row, col] entry of the matrix
	double get(lapack_int row, lapack_int col) const;

	//pass by value the [k] entry of the components array
	double get(lapack_int k) const;
	/*	Do Matrix-Matrix Multiplication. If dims don't match, return an
	 *	empty matrix via the Default Constructor;
	 */
	Matrix operator*(const Matrix& B) const;

	/*
	 *	modal multiplication of this matrix and a tensor T:
	 *	Take in a tensor which has the same mu-index length as the number of
	 *	columns in this matrix. The compute a new tensor in the following way:
	 *
	 *	K(i_1, ... , j , ... , i_end)
	 *			= sum( i_mu = 1, i_mu = end,
	 *				this->entry(j, i_mu) * T(i_1, ... , i_mu , ... , i_end) )
	 *	
	 */
	Tensor modalProd(lapack_int mu, const Tensor& T) const;	
	
	/*
	 *	Make a call to Tensor::operator=().
	 */
	Matrix& operator=(const Tensor& B);

	/*
	 * Due to a constraint of C++ operator overloading, we can only do
	 * right hand scalar multiplication. This isn't a problem though, since
	 * scalars are defined to commute with a matrix.
	 */
	Matrix operator*(const double a) const;

	//component-wise division
	Matrix operator/(const double a) const;

	/*
	 * Matrix-double addition operator. 
	 */
	Matrix operator+(const double a) const;

	/*
	 * Matrix-double subtraction operator. 
	 */
	Matrix operator-(const double a) const;

	/*
	 * Matrix-Matrix addition operator. 
	 */
	Matrix operator+(const Matrix& B) const;

	/*
	 * Matrix-Matrix subtraction operator. 
	 */
	Matrix operator-(const Matrix& B) const;

	/*
	 * Standard inner product of two vectors. called by cblas's ddot routine.
	 */
	double dot(const Matrix& x) const;

	/*
	 * Takes the sum of squares of entries of this matrix. 
	 */
	double norm2() const;
	
	/*
	 * Swap rows r1 and r2
	 */
	void swapRows(lapack_int r1, lapack_int r2);

	/*
	 * Scale the Row with index r
	 */
	void scaleRow(lapack_int r, double scaleBy);

	/*
	 *	Use LAPACKE QR algorithm to get Q and R. Computes reduced QR.
	 *	i.e. compute (*this) = Q*R
	 */
	void qr(Matrix& Q, Matrix& R) const;

	/*
	 *	Use LAPACKE dgees algorithm to compute a Schur decomposition
	 *	of this matrix object. If this matrix object commutes with its
	 *	transpose, then the schur form is unique up to a permutation of the
	 *	eigenblocks. (This is a restatement of the spectral theorem.)
	 *
	 *	The eigenblocks are 2x2 for the case of complex eigenvalues. This means
	 *	that the schur form is a safe to compute a decomposition for a matrix
	 *	with complex eigenvalues.
	 *
	 *	I.e. compute (*this) = Z * S * Z.transp()
	 */
	void schur(Matrix& Z, Matrix& R) const;

	/*
	 *	Use LAPACKE dgesvd algorithm to get the singular value decomposition.
	 *	I.e. compute (*this) = U * Sig.diag() * Vt
	 */
	void svd(Matrix& Q, Matrix& Sig, Matrix& Vt) const;

	/*
	 *	Use LAPACKE dgesvdx algorithm to get the first numVectors left signular
	 *	vectors of this matrix object. Return the result. If you ask for
	 *	more vectors than exist, we return all the singular vectors.
	 */
	Matrix getLeftSV(lapack_int numVectors) const;

	/*
	 *	Use personally written linear solver. (Gauss Jordan with pivoting)
	 *	Input: a matrix Object.
	 *	Return: Matrix solution to the problem: A*X = B
	 */
	Matrix gjSolve(const Matrix& B) const;

	/*
	 * Use LAPACKE dgelsy to compute the min norm least square solution to:
	 * A*X = B
	 */
	Matrix lsSolve(const Matrix& B) const;

	/*
	 *	An implementation of the matlab/octave diag(M) function.
	 *	If this object is a row or column vector (i.e. num components 
	 *	is equal to num rows or num cols,) then the function
	 *	returns a matrix with this vector on the diagonal. Otherwise,
	 *	this function returns a vector of size (MIN(numRows,numCols) , 1)
	 *	with this->get(i,i) on component i
	 */
	Matrix diag() const;

	/*
	 * Computes the sum of the diagonal as defined by diag().
	 */
	double trace() const;
	/*
	 *	Matrix transpose. Returns the transpose of this matrix.
	 */
	Matrix transp() const;

	/*
	 * Kronecker product of matrices
	 */
	Matrix kron(const Matrix& B) const;
	/*
	 * Hadamard Product of Matrices. This is equivalent to ".*" in Matlab
	 */
	Matrix hada(const Matrix& B) const;

	/*
	 * Concatenate 2 matrices. This is the same as creating an "augmented"
	 * matrix. in matlab syntax: A.cat(B) = [A, B].
	 */
	Matrix cat(const Matrix& B) const;

	/*
	 * Get the r^th row as a row vector. (1 row, multiple columns)
	 */
	Matrix getRow(lapack_int r) const;

	/*
	 * Get the c^th column as a column vector. (multiple rows, 1 column)
	 */
	Matrix getCol(lapack_int c) const;

	lapack_int getNumRows() const;
	lapack_int getNumCols() const;

	//Print Matrix to terminal
	void print(void) const;
	/*
	 * Print Matrix to a file in a format which can be loaded into matlab/octave 
	 *
	 * First argument is the matlab variable name and second argument is the 
	 * file name to save it as.
	 */
	void matlabSave(char *varName, char *fileName) const;
	
	/*
	 * A data loading function which skips the first line of the file
	 * and special character " ],;" defined by Matlab. This loader is compatible
	 * with matlabSave().
	 */
	void matlabLoad(char *fileName);
};

//};//end namespace moctalab

#endif
