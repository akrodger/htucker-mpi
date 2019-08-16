/*
 * Dense Tensor Object implementation by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include"Tensor.h"
#ifndef NULL
	#define NULL
#endif


//using namespace moctalab;

/*
private:
	double *components;
	lapack_int *indexLength;
	lapack_int dim;
*/
/*
 * Locally used helper functions:
 */

/*
 * Static Local Variables:
 */

/*
 * Function Implementations:
 */


Tensor::Tensor(){
	this->dim = 0;
	this->numComponents = 0;
}
/*
 * sizeVector is array of max indexes. dimension is the number of elements
 * in sizeVector.
 */
Tensor::Tensor(std::valarray<lapack_int>& sizeVector, lapack_int dimension){
	this->init(sizeVector, dimension);
}

Tensor::Tensor(const Tensor& copyFrom){
	lapack_int i;
	this->init(copyFrom.indexLength, copyFrom.dim);
	for(i = 0; i < copyFrom.getNumComponents(); i++){
		this->components[i] = copyFrom.components[i];
	}
}

Tensor::~Tensor(){
}

lapack_int Tensor::init(const std::valarray<lapack_int>& sizeVector,
						lapack_int dimension){
	lapack_int totalElements = 1;
	lapack_int i = 0;
	this->indexLength = sizeVector;
	/*		//(lapack_int*) malloc(sizeof(lapack_int)*dimension);*/
	while(i < dimension){
		totalElements *= sizeVector[i];
		//this->indexLength[i] = sizeVector[i];
		i++;
	}
	this->dim = dimension;
	this->numComponents = totalElements;
	this->components.resize(totalElements);
	// = (double*) malloc(sizeof(double)*totalElements);
	return 0;
}
/*
 * Return the Value located at the given multiIndex.
 */
double& Tensor::entry(std::valarray<lapack_int>& multiIndex){
	register lapack_int k, l, memOffset, product, d;
	//Iterate through the multi-index in column major order
	memOffset = multiIndex[0];
	k = 1;
	d = dim;
	product = 1;
	while(k < d){
		l = k-1;
		product *= indexLength[l];
		memOffset += multiIndex[k] * product;
		k++;		
	}
	return components[memOffset];
}

double& Tensor::operator()(std::valarray<lapack_int>& multiIndex){
	register lapack_int k, l, memOffset, product, d;
	//Iterate through the multi-index in column major order
	memOffset = multiIndex[0];
	k = 1;
	d = dim;
	product = 1;
	while(k < d){
		l = k-1;
		product *= indexLength[l];
		memOffset += multiIndex[k] * product;
		k++;		
	}
	return components[memOffset];
}

double Tensor::get(std::valarray<lapack_int>& multiIndex) const{
	register lapack_int k, l, memOffset, product, d;
	//Iterate through the multi-index in column major order
	memOffset = multiIndex[0];
	k = 1;
	d = dim;
	product = 1;
	while(k < d){
		l = k-1;
		product *= indexLength[l];
		memOffset += multiIndex[k] * product;
		k++;
	}
	return components[memOffset];
}

double Tensor::get(lapack_int k) const{
	return this->components[k];
}
//Assignment/Copy Operator
Tensor& Tensor::operator=(const Tensor& B){
	this->indexLength = B.indexLength;
	this->components = B.components;
	this->dim = B.dim;
	this->numComponents = B.numComponents;
	return *this;
}

Tensor Tensor::operator*(const double a) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components * a;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::operator/(const double a) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components / a;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::operator+(const Tensor& B) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components + B.components;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::operator+(const double a) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components + a;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::operator-(const Tensor& B) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components - B.components;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::operator-(const double a) const{
	Tensor res;
	res.indexLength = this->indexLength;
	res.components = this->components - a;
	res.dim = this->dim;
	res.numComponents = this->numComponents;
	return res;
}

Tensor Tensor::diagCat(const Tensor& B) const{
	lapack_int i, j;
	std::valarray<lapack_int> multiIndex, B_index;
	Tensor C = Tensor();
	if(this->dim != B.dim){
		return C;
	}
	C.dim = this->dim;
	C.indexLength.resize(C.dim);
	C.numComponents = 1;
	for(i = 0; i < C.dim; i++){
		C.indexLength[i] = this->indexLength[i] + B.indexLength[i];
		C.numComponents *= C.indexLength[i];
	}
	C.components.resize(C.numComponents);
	multiIndex.resize(C.dim);
	B_index.resize(B.dim);
	C.components = 0;
	multiIndex = 0;
	B_index = 0;
	for(i = 0; i < this->numComponents; i++){
		C(multiIndex) = this->get(multiIndex);
		C.multiIterate(multiIndex, this->indexLength, NULL, this->dim);
	}
	for(i = 0; i < B.numComponents; i++){
		for(j = 0; j < C.dim; j++){
			if(multiIndex[j] == 0){
				multiIndex[j] = this->indexLength[j];
			}
		}
		C(multiIndex) = B.get(B_index);
		C.multiIterate(multiIndex, C.indexLength, NULL, C.dim);
		B.multiIterate(B_index, B.indexLength, NULL, B.dim);
	}
	return C;
}

lapack_int Tensor::getIndexLength(lapack_int i) const{
	return this->indexLength[i];
}

lapack_int Tensor::getNumComponents() const{
	return this->numComponents;
}

lapack_int Tensor::getDim() const{
	return dim;
}


void Tensor::multiIterate(	std::valarray<lapack_int>& iter,
							const std::valarray<lapack_int>& indexBound,
							const std::valarray<lapack_int>& iterSubset,
							lapack_int subsetSize) const{
	lapack_int i;
	lapack_int counter = 0;
	lapack_int recurse = 1;
	while(counter < subsetSize && recurse == 1){
		recurse = 0;
		i=counter;
		//i = subsetSize - counter - 1;
		iter[iterSubset[i]] += 1;
		//this case handles carrying over.
		if(iter[iterSubset[i]] >= indexBound[iterSubset[i]]){
			iter[iterSubset[i]] = 0;
			counter++;
			recurse = 1;
		}
	}
}

void Tensor::multiIterate(	std::valarray<lapack_int>& iter,
							const std::valarray<lapack_int>& indexBound,
							void*,
							lapack_int iterSize) const{
	lapack_int i;
	lapack_int counter = 0;
	lapack_int recurse = 1;
	while(counter < iterSize && recurse == 1){
		recurse = 0;
		i=counter;
		//i = subsetSize - counter - 1;
		iter[i] += 1;
		//this case handles carrying over.
		if(iter[i] >= indexBound[i]){
			iter[i] = 0;
			counter++;
			recurse = 1;
		}
	}
}

//end namespace moctalab
