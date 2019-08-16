/*
 * Dense Matrix Object implementation by Bram Rodgers.
 * Prior to LGPL Draft Dated: 3, Sept 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include"Matrix.h"
#ifndef NULL
	#define NULL
#endif

//using namespace moctalab;

/*
 * Locally used helper functions:
 */


/*
 * Static Local Variables:
 */

/*
 * Function Implementations:
 */

Matrix::Matrix() : Tensor(){};

Matrix::Matrix(lapack_int numRows, lapack_int numCols) : Tensor(){
	std::valarray<lapack_int> sizeVector(2);
	sizeVector[0] = numRows;
	sizeVector[1] = numCols;
	this->Tensor::init(sizeVector, 2);
}

Matrix::Matrix(lapack_int numRows, lapack_int numCols, char initType){
	lapack_int i, j;
	std::valarray<lapack_int> sizeVector(2);
	sizeVector[0] = numRows;
	sizeVector[1] = numCols;
	this->Tensor::init(sizeVector, 2);
	switch(initType){
		case '0':
				for(i = 0; i < this->getNumComponents(); i++){
					this->components[i] = 0; 
				}
				break;
		case '1':
				for(i = 0; i < this->getNumComponents(); i++){
					this->components[i] = 1; 
				}
				break;
		case 'I':
				for(j = 0; j < numCols; j++){
					for(i = 0; i < numRows; i++){
						if(i == j){
							this->entry(i,j) = 1;
						}else{
							this->entry(i,j) = 0;
						}
					}
				}
				break;
		case 'U':
				for(j = 0; j < numCols; j++){
					for(i = 0; i < numRows; i++){
						this->entry(i,j) =  rand() / (RAND_MAX + 1.0);
					}
				}
				break;
		default:
				printf("\nDeclare Error: Char had type"
						" not in declaration cases.\n"
						"\nChar used: %c\n\n", initType);
				exit(1);
				break;
	}
}



Matrix::Matrix(const Tensor& T, std::valarray<lapack_int>& rowIndex,
				lapack_int rowIndexSize){

	lapack_int numRows, numCols, i, j, rowIndexFlag, colStoreIndex,
					numComponents = T.getNumComponents();
	std::valarray<lapack_int> sizeVector(2);

	lapack_int d = T.getDim();
	std::valarray<lapack_int> iter(d);
	std::valarray<lapack_int> indexBoundOfT(d);
//	lapack_int colIndexSize; 
//	lapack_int *colIndex;
	std::valarray<lapack_int> permutedIndex(d);
	std::valarray<lapack_int> memWeight(d);
	lapack_int memOffset;
	lapack_int recurse = 1;
	lapack_int counter = 0;
	numRows = 1;
	i = 0;

	while(i < rowIndexSize){//Find how many rows there will be.
		numRows *= T.getIndexLength(rowIndex[i]);
		i++;
	}

	//Apply the matricization size identity to get the number of columns
	numCols = numComponents / numRows; 
	//Allocate memory for the column multiIndex.
//	colIndexSize = d - rowIndexSize;
//	colIndex = (lapack_int*) malloc(colIndexSize*sizeof(lapack_int));
	colStoreIndex = 0;
	i = 0;
	while(i < d){ //Construct index bounds and column multiIndex
		indexBoundOfT[i] = T.getIndexLength(i);
		iter[i] = indexBoundOfT[i];
		j = 0;
		rowIndexFlag = 0;
		while(j < rowIndexSize){ //check if i is a row index
			if(i == rowIndex[j]){ // if it is, put up a flag
				permutedIndex[j] = i;
				rowIndexFlag = 1;
				break;
			}
			j++;
		}
		if(rowIndexFlag == 0){ // if index i is not a column index
//			colIndex[colStoreIndex] = i; // then store it
			permutedIndex[colStoreIndex + rowIndexSize] = i;
			colStoreIndex++;
		}
		rowIndexFlag = 0; //end of this iteration, put flag down.
		i++;
	}
//	for(i=0;i<d;i++){
//		printf("\nindexBoundOfT[%ld] = %ld\n", i, indexBoundOfT[i]);
//	}
//	for(i=0;i<rowIndexSize;i++){
//		printf("\nrowIndex[%ld] = %ld\n", i, rowIndex[i]);
//	}
//	printf("\n#####################\n");
//	for(i=0;i<colIndexSize;i++){
//		printf("\ncolIndex[%ld] = %ld\n", i, colIndex[i]);
//	}
//	printf("\n&&&&&&&&&&&&&&&&&&&&&\n");
//	for(i=0;i<d;i++){
//		printf("\npermutedIndex[%ld] = %ld\n", i, permutedIndex[i]);
//	}
//	printf("\n_____________________\n");
	sizeVector[0] = numRows;
	sizeVector[1] = numCols;
	this->Tensor::init(sizeVector, 2); //initialize the memory of the matrix
//	Matrix L = Matrix(*this);
//	Matrix M = Matrix(*this);

//	for(i = 0; i < d; i++){
//		iter[i] = 0;
//	}
//	j = 0;
//	while(j < numCols){
//		i = 0;
//		while(i < numRows){
//			this->components[i+(numRows*j)] = T.get(iter);
//			Tensor::multiIterate(iter, indexBoundOfT, 
//								rowIndex, rowIndexSize);
//			i++;
//		}
//		Tensor::multiIterate(iter, indexBoundOfT, 
//							colIndex, colIndexSize);
//		j++;
//	}


	for(i = 0; i < d; i++){
		//calculate memory offset weights:
		if(i == 0){
			memWeight[i] = 1;
		}else{
			memWeight[i] = indexBoundOfT[i-1]*memWeight[i-1];
		}
	}
//	for(i=0;i<d;i++){
//		printf("\nindexBoundOfT[%ld] = %ld\n", i, indexBoundOfT[i]);
//	}
//	printf("\n&&&&&&&&&&&&&&&&&&&&&\n");
	for(i = 0; i < d; i++){
		indexBoundOfT[i] = iter[permutedIndex[i]];
	}
	for(i = 0; i < d; i++){
		iter[i] = 0;
	}
//	for(i=0;i<d;i++){
//		printf("\npermutedIndex[%ld] = %ld\n", i, permutedIndex[i]);
//	}
//	printf("\n_____________________\n");
//	for(i=0;i<d;i++){
//		printf("\nindexBoundOfT[%ld] = %ld\n", i, indexBoundOfT[i]);
//	}
	for(i = 0; i < numComponents; i++){
		memOffset = 0;
		for(j = 0; j < d; j++){
			memOffset += iter[j]*memWeight[permutedIndex[j]];
		}
		this->components[i] = T.components[memOffset];
		recurse = 1;
		counter = 0;
		//j = rowIndexSize - 1;
		j = 0;
		while(counter < d && recurse == 1){
			recurse = 0;
			//i = subsetSize - counter - 1;
			iter[j] += 1;
			//this case handles carrying over.
			if(iter[j] >= indexBoundOfT[j]){
				iter[j] = 0;
				recurse = 1;
				j += 1;
				//if(j < 0){j=d-1;}
			}
			counter++;
		}
	}

//	matricize(	T.components,
//				indexBoundOfT,
//				&d,
//				this->components,
//				permutedIndex,
//				&numComponents);
//	printf("\nCorrect:");
//	this->print();
//	printf("\nC Reshape:");
//	M.print();
//	printf("\nDifference:");
//	Matrix N = M - *this;
//	N.print();
//	printf("\nFortran Reshape:");
//	L.print();

//	free(colIndex);

}


Matrix::Matrix(const Matrix& copyFrom) : Tensor(){
	lapack_int i;
	std::valarray<lapack_int> sizeVector(2);
	sizeVector[0] = copyFrom.getNumRows();
	sizeVector[1] = copyFrom.getNumCols();
	this->Tensor::init(sizeVector, 2);
	for(i = 0; i < copyFrom.getNumComponents(); i++){
		this->components[i] = copyFrom.components[i];
	}
}

Matrix::~Matrix(){
}

double& Matrix::entry(lapack_int row, lapack_int col){
	return this->components[row + col*Tensor::getIndexLength(0)];
}

double& Matrix::operator()(lapack_int row, lapack_int col){
	return this->components[row + col*Tensor::getIndexLength(0)];
}

double& Matrix::operator()(lapack_int index){
	return this->components[index];
}

double Matrix::get(lapack_int row, lapack_int col) const{
	return this->components[row + col*Tensor::getIndexLength(0)];
}

double Matrix::get(lapack_int k) const{
	return this->Tensor::get(k);
}

Tensor Matrix::modalProd(lapack_int mu, const Tensor& T) const{
	lapack_int 	i = 0, j = 0, d = T.getDim(), 
			numCols = this->getNumCols(), numRows = this->getNumRows();
	std::valarray<lapack_int> multiIndex(d);
	std::valarray<lapack_int> sizeVector(d);
	std::valarray<lapack_int> iterSubset(d-1);
	std::valarray<lapack_int> rowIndex(1);
	lapack_int colIterates;
	rowIndex[0] = mu;
	Matrix matOfT = Matrix(T, rowIndex, 1);
	//Set up multi-iterator and memory size for output
	for(i = 0; i < d; i++){
		if(i < mu){
			iterSubset[i] = i;
		}else if(i < d-1){
			iterSubset[i] = i+1; 
		}
		multiIndex[i] = 0;
		if(i != mu){
			sizeVector[i] = T.getIndexLength(i);
		}else{
			sizeVector[i] = this->getNumRows();
		}
	}
	//define iteration based on column entry.
	colIterates = matOfT.getNumCols();
	Tensor P = Tensor(sizeVector, d);
	Matrix prod = Matrix(numRows, colIterates);
	if(mu < d){
		if(numCols == T.getIndexLength(mu)){
			//multiply by matricization
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							numRows, colIterates,
							numCols, 1.0, &(this->components[0]),
							numRows, &(matOfT.components[0]), numCols,
							0, &(prod.components[0]), numRows);
			//Store product into new tensor
			for(j = 0; j < colIterates; j++){
				multiIndex[mu] = 0;
				for(i = 0; i < numRows; i++){
					P.entry(multiIndex) = 
									prod.components[i + j*numRows];
					(multiIndex[mu])++;
				}
				P.multiIterate(multiIndex, sizeVector, iterSubset, d-1);
			}
		}else{
			printf("\nError: Attempting to multiply into mismatched mode"
					" lengths."
					"\nmu  = %ld,"
					"\nLength of this matrix's cols      = %ld"
					"\nLength of input Tensor's mu index = %ld\n", 
					mu, this->getNumCols(), T.getIndexLength(mu));
			exit(1);
		}
	}else{
		printf("\nError: Attempting to multiply into mode mu greater"
				" than the dimension of the given Tensor."
				"\nmu  = %ld,\n dim = %ld\n", mu, d);
		exit(1);
	}
	return P;
}


Matrix& Matrix::operator=(const Tensor& B){
	if(B.getDim() == 2){
		((Tensor*)this)->operator=(B);
	}
	return *this;
}

Matrix Matrix::operator*(const Matrix& B) const{
	lapack_int this_rows, this_cols, B_rows, B_cols;
	this_rows = this->getNumRows();
	this_cols = this->getNumCols();
	B_rows = B.getNumRows();
	B_cols = B.getNumCols();
	Matrix prod = Matrix(this_rows, B_cols);
	if(B_rows != this_cols){
		printf("\nMatrix Mult Error: Row and column length mismatch.\n"
				"Left was size  : %ld x %ld\n"
				"Right was size : %ld x %ld\n",
				this_rows, this_cols, B_rows, B_cols);
		exit(1);
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				this_rows, B_cols,
				this_cols, 1.0, &(this->components[0]),
				this_rows, &(B.components[0]), B_rows,
				0, &(prod.components[0]), this_rows);
	return prod;
}

Matrix Matrix::operator*(const double a) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator*(a);
	return res;
}

Matrix Matrix::operator/(const double a) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator/(a);
	return res;
}

Matrix Matrix::operator+(const double a) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator+(a);
	return res;
}

Matrix Matrix::operator-(const double a) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator-(a);
	return res;
}

Matrix Matrix::operator+(const Matrix& B) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator+(B);
	return res;
}

Matrix Matrix::operator-(const Matrix& B) const{
	Matrix res = Matrix();
	res = ((Tensor*)this)->operator-(B);
	return res;
}

double Matrix::dot(const Matrix& x) const{
	int N = (int) x.getNumComponents();
	if(this->getNumComponents() != x.getNumComponents()){
		printf("\nDot Product Error: Array length mismatch.\n"
				"Left was size  : %ld\n"
				"Right was size : %d\n",
				this->getNumComponents(), N);
		exit(1);
	}
	return cblas_ddot(N, &(this->components[0]), 1, &(x.components[0]), 1);
}

double Matrix::norm2() const{	
	int N = (int) this->getNumComponents();
	return cblas_dnrm2(N, &(this->components[0]), 1);
}

void Matrix::qr(Matrix& Q, Matrix& R) const{
	lapack_int k, rDiagSize, i, j;
	std::valarray<double> qComponents;
	Q = *this;
	k = MIN( Q.getNumRows(), Q.getNumCols());
	double *tau = (double*) malloc(k*sizeof(double));
	double diagSign;
	LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q.getNumRows(), Q.getNumCols(),
				&(Q.components[0]), Q.getNumRows(), tau);
	R = Matrix(k, Q.getNumCols(), '0');
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', R.getNumRows(), R.getNumCols(),
					&(Q.components[0]), Q.getNumRows(), 
					&(R.components[0]), R.getNumRows());
	LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.getNumRows(), k,
					k, &(Q.components[0]), Q.getNumRows(), tau );
	if(k != Q.getNumCols()){
		qComponents.resize(Q.getNumRows()*k);
		for(k = 0; k < Q.getNumRows()*R.getNumRows(); k++){
			qComponents[k] = Q.components[k];
		}
		Q = Matrix(Q.getNumRows(), R.getNumRows());
		Q.components = qComponents;
	}
	rDiagSize = MIN(R.getNumRows(), R.getNumCols());
	for(j = 0; j < rDiagSize; j++){
		if(R(j,j) != 0){
			diagSign = R(j,j)/ABS(R(j,j));
		}else{
			diagSign = 1.0;
		}
		if(diagSign < 0.0){
			for(i = j; i < R.getNumCols(); i++){
				R(j,i) *= diagSign;
			}
			for(i = 0; i < Q.getNumRows(); i++){
				Q(i,j) *= diagSign;
			}
		}
	}
	free(tau);
}

//A routine which serves only the purpose of giving select in the schur
//decomposition call a definition. Could be changed to give an ordering
//to the schur decomp.
lapack_logical schurSelection(const double *re, const double *im){
	return 0;
}

void Matrix::schur(Matrix& Z, Matrix& R) const{
	char jobvs = 'V', sort = 'N';
	LAPACK_D_SELECT2 select = &schurSelection;
	lapack_int m = this->getNumRows();
	lapack_int n = this->getNumCols();
	lapack_int sdim = 0;
	std::valarray<double> wReal;
	std::valarray<double> wImag;
	if(m != n){
		printf("\nSchur Decomp Error: This is not a square matrix.\n"
		"Size was : %ld x %ld\n",
		m, n);
		exit(1);
	}
	wReal.resize(m);
	wImag.resize(m);
	R = *this;
	Z = Matrix(m,m);
	LAPACKE_dgees(	LAPACK_COL_MAJOR, jobvs, sort,
					select, m, 
					&(R.components[0]),
					m, &sdim, &(wReal[0]),
					&(wImag[0]), &(Z.components[0]), m);
}

void Matrix::svd(Matrix& U, Matrix& Sig, Matrix& Vt) const{
	lapack_int m = this->getNumRows(), n = this->getNumCols();
	lapack_int lwork = MAX(1,5*MIN(m,n)) + MAX(1,3*MIN(m,n));
	lapack_int lapack_info;
	std::valarray<double> work(lwork);
	char jobType = 'A';
	Matrix copyThis = Matrix(*this);
	Sig = Matrix(MIN(m,n), 1);
	U = Matrix(m,m);
	Vt = Matrix(n,n);
	LAPACK_dgesvd( &jobType, &jobType, &m, &n,
                    &(copyThis.components[0]), &m, 
					&(Sig.components[0]), &(U.components[0]), &m,
                    &(Vt.components[0]), &n, &(work[0]), &(lwork),
					&lapack_info );
}

Matrix Matrix::getLeftSV(lapack_int numVectors) const{
	lapack_int m = this->getNumRows(), n = this->getNumCols(),
					ldvt = 1, i, j, min_m_n;
	lapack_int copyNum = MIN(numVectors, MIN(m,n));
	lapack_int numFound = copyNum;
	lapack_int superbSize = 12*MIN(m,n);
	char jobu = 'v', jobvt = 'N', range = 'I';
	Matrix copyThis = Matrix(*this);
	Matrix U_trunc = Matrix(m, copyNum);
	min_m_n = 2*MIN(m,n);
	double *Sig = (double*) malloc(sizeof(double)*min_m_n);
	double *Ucomponents = (double*) malloc(sizeof(double)*m*copyNum);
	double *Vtcomponents = (double*) malloc(sizeof(double)*ldvt);
	lapack_int *superb = (lapack_int*) malloc(sizeof(lapack_int)*superbSize);
	LAPACKE_dgesvdx( LAPACK_COL_MAJOR, jobu, jobvt, range,
                           m, n, &(copyThis.components[0]),
                           m, 0, 0,
                           1, copyNum, &numFound,
                           Sig, Ucomponents, m, Vtcomponents, ldvt, superb);
	for(j = 0; j < copyNum; j++){
		for(i = 0; i < m; i++){
			U_trunc(i,j) = Ucomponents[j*m + i];
		}
	}
	free(Ucomponents);
	free(Sig);
	free(Vtcomponents);
	free(superb);
	return U_trunc;
}


void Matrix::swapRows(lapack_int r1, lapack_int r2){
	lapack_int j=0, tot = this->getNumCols();
	double temp;
	Matrix *T = this;/*,chunkSize;
	#pragma omp parallel shared(T, tot)\
						private(j, temp, chunkSize)
	{
		chunkSize = tot / ((lapack_int) omp_get_num_threads());
		chunkSize++;
		#pragma omp for schedule(dynamic,chunkSize)
	*/
		for (j = 0; j < tot; j++) {
			temp = T->entry(r1,j);
			T->entry(r1,j) = T->entry(r2,j);
			T->entry(r2,j) = temp;
		}
	//}
	return;
}

void Matrix::scaleRow(lapack_int r, double scaleBy){
	lapack_int j=0, tot = this->getNumCols();
	Matrix *T = this;/* chunkSize;
	#pragma omp parallel shared(T, tot)\
						private(j, chunkSize)
	{
		chunkSize = tot / ((lapack_int) omp_get_num_threads());
		chunkSize++;
		#pragma omp for schedule(dynamic,chunkSize)
	*/
		for (j = 0; j < tot; j++) {
			T->entry(r,j) *= scaleBy;
		}
	//}
	return;
}

/*
 * Use personally written linear solver. (Gauss Jordan with pivoting)
 * Input: a matrix Object.
 * Return: Matrix solution to the problem: A*X = B
 */
Matrix Matrix::gjSolve(const Matrix& B) const{
	lapack_int h = 0, k = 0, i = 0, j = 0, x = 0, y = 0, iMax = 0;
	lapack_int numCols = this->getNumCols();
	lapack_int m = this->getNumRows();
	lapack_int n = this->getNumCols() + B.getNumCols();
	//first find the first nonzero row from the bottom up:
	//this is the thing that tells us we found a pivot column
	double f = 0;
	Matrix A = Matrix(m, n);
	Matrix C = Matrix(m, B.getNumCols());
	const Matrix *T = this;
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			if(j < numCols){
				A(i,j) = T->get(i,j);
			}else{
				A(i,j) = B.get(i, j - numCols);
			}
		}
	}
	while(h < m && k < n){
		//Find the maximal row index:
		iMax = h;
		for(i = h; i < m; i++){
				if(ABS(A(i,k)) > ABS(A(iMax,k))){
					iMax = i;
				}
		}
		//If there is no pivot, then continue:
		if(ABS(A(iMax,k)) < EPS_M){
			k++;
		}else{
			//Swap rows with iMax
			if(iMax != h){
				for(j = k; j < n; j++){
					f = A(h,j);
					A(h,j) = A(iMax,j);
					A(iMax,j) = f;
				}
			}
			f = A(h,k);
			//Scale the current Row:
			for(j = k; j < n; j++){
				A(h,j) = A(h,j)/f;
			}
			//Eliminate the rows above and below the pivot:
			for(i = 0; i < m; i++){
				if(i != h){
					f = A(i, k);
					A(i,k) = 0;
					for(j = k+1; j < n; j++){
						A(i,j) = A(i,j) - f*A(h,j);
					}
				}
			}
			h++; k++;
		}
	}
	//Fill the matrix C with the solution.
	x = C.getNumRows();
	y = C.getNumCols();
	k = this->getNumCols();
	for(i = 0; i < x; i++){
		for(j = 0; j < y; j++){
			C(i,j) = A.get(i, j + k);
		}
	}
	return C;
}

Matrix Matrix::lsSolve(const Matrix& B) const{
	lapack_int m, n, nrhs, ldb, rank = 0, i, j;
	Matrix thisFactored = *this;
	Matrix Solution;
	std::valarray<double> bComponents;
	std::valarray<lapack_int> jpvt;
	bComponents = B.components;
	m = thisFactored.getNumRows();
	n = thisFactored.getNumCols();
	ldb = B.getNumRows();
	nrhs = B.getNumCols();
	Solution = Matrix(n, nrhs);
	jpvt.resize(n);
	if(m != ldb){
		printf("\nLeast Square Solution Error: LHS numRows not equal RHS's.\n"
				"LHS was size : %ld x %ld\n"
				"RHS was size : %ld x %ld\n",
				m, n, ldb, nrhs);
		exit(1);
	}
	LAPACKE_dgelsy(LAPACK_COL_MAJOR,
					m,
					n,
					nrhs,
					&(thisFactored.components[0]),
					m,
					&(bComponents[0]),
					ldb, 
					&(jpvt[0]),
					 0,
					&rank);
	for(j = 0; j < nrhs; j++){
		for(i = 0; i < n; i++){
			Solution(i,j) = bComponents[i + (n*j)];
		}
	}
	return Solution;
}

Matrix Matrix::diag() const{
	Matrix D;//the return variable
	lapack_int numR, numC, numE,//numRows, numCols, numElements of vector
				k, j;//loop iterators
	numE = this->getNumComponents();
	numR = this->getNumRows();
	numC = this->getNumCols();
	if((numR == numE) || (numC == numE)){//if this is a col or row vector
		D = Matrix(numE, numE);
		for(k = 0; k < numE; k++){
			for(j = 0; j < numE; j++){
				if(j == k){
					D(j,k) = this->get(j);
				}else{
					D(j,k) = 0;
				}
			}
		}
	}else{
		numE = MIN(numR, numC);
		D = Matrix(numE, 1);
		for(k = 0; k < numE; k++){
			D(k) = this->get(k,k);
		}
	}
	return D;
}

double Matrix::trace() const{
	return this->diag().components.sum();
}

Matrix Matrix::transp() const{
	Matrix A = Matrix(this->getNumCols(), this->getNumRows());
	lapack_int i, j;
	for(i = 0; i < this->getNumRows(); i++){
		for(j = 0; j < this->getNumCols(); j++){
			A(j,i) = this->get(i,j);
		}
	}
	return A;
}

Matrix Matrix::kron(const Matrix& B) const{
	lapack_int r1 = this->getNumRows();
	lapack_int r2 = B.getNumRows();
	lapack_int c1 = this->getNumCols();
	lapack_int c2 = B.getNumCols();
	lapack_int r1r2 = r1*r2;
	lapack_int c1c2 = c1*c2;
	lapack_int i1, i2, j1, j2, a, b, iter;
	lapack_int tot = r1r2*c1c2;
	Matrix K = Matrix(r1r2, c1c2);
	const Matrix *A = this;
	/*
	lapack_int chunkSize;
	#pragma omp parallel shared(K, A, B, r1, r2, c1, c2, r1r2, c1c2, tot)\
						private(iter, i1, i2, j1, j2, a, b, chunkSize)
	{
		chunkSize = tot / ((lapack_int) omp_get_num_threads());
		chunkSize++;
		#pragma omp for schedule(dynamic,chunkSize)
	*/
		for(iter = 0; iter < tot; iter++){
			b = iter/r1r2;
			a = iter - b*r1r2;
			i1 = a/r2;
			i2 = a - (i1*r2);
			j1 = b/c2;
			j2 = b - (j1*c2);
			K(iter) = A->get(i1,j1) * B.get(i2,j2);
		}
	//}
	return K;
}

Matrix Matrix::hada(const Matrix& B) const{
	Matrix H = Matrix(B.getNumRows(), B.getNumCols());
	lapack_int i, N = H.getNumComponents();
	if( (H.getNumCols() == this->getNumCols()) &&
		(H.getNumRows() == this->getNumRows())){
		for(i = 0; i < N; i++){
			H.components[i] = (B.components[i]) * (this->components[i]);
		}
	}else{
		printf("Hadamard Error: Matrix Dimensions Must Agree.\n"
				"This was : %ld x %ld\n"
				"B was    : %ld x %ld\n",
				this->getNumRows(), this->getNumCols(),
				B.getNumRows(), B.getNumCols());
		exit(1);
	}
	return H;
}

Matrix Matrix::getRow(lapack_int r) const{
	lapack_int numCols = this->getNumCols();
	lapack_int numRows = this->getNumRows();
	Matrix rowVec = Matrix(1, numCols);
	lapack_int j;
	/*
	lapack_int chunkSize;
	#pragma omp parallel shared(numRows,numCols) private(j, chunkSize)
	{
		chunkSize = numCols / ((lapack_int) omp_get_num_threads());
		chunkSize++;
		#pragma omp for schedule(dynamic,chunkSize)
	*/
		for(j = 0; j < numCols; j++){
			rowVec.components[j] = this->components[r + (numRows*j)];
		}
	//}
	return rowVec;
}

Matrix Matrix::getCol(lapack_int c) const{
	lapack_int numRows = this->getNumRows();
	Matrix colVec = Matrix(numRows, 1);
	lapack_int i;
	/*
	lapack_int chunkSize;
	#pragma omp parallel shared(numRows) private(i, chunkSize)
	{
		chunkSize = numRows / ((lapack_int) omp_get_num_threads());
		chunkSize++;
		#pragma omp for schedule(dynamic,chunkSize)
	*/
		for(i = 0; i < numRows; i++){
			colVec.components[i] = this->components[i + (c*numRows)];
		}
	//}
	return colVec;
}

Matrix Matrix::cat(const Matrix& B) const{
	lapack_int i, j;
	lapack_int numRows = this->getNumRows();
	lapack_int numCols = this->getNumCols();
	lapack_int catNumCols = this->getNumCols() + B.getNumCols();
	Matrix C = Matrix(numRows, catNumCols);
	if(numRows == B.getNumRows()){
		for(j = 0; j < catNumCols; j++){
			for(i = 0; i < numRows; i++){
				if(j < numCols){
					C.components[i + numRows*j] = 
											this->components[i + numRows*j];
				}else{
					C.components[i + numRows*j] = 
										B.components[i + numRows*(j-numCols)];
				}
			}
		}
	}
	return C;
}

lapack_int Matrix::getNumRows() const{return Tensor::getIndexLength(0);}

lapack_int Matrix::getNumCols() const{return Tensor::getIndexLength(1);}

void Matrix::print(void) const{
	lapack_int i, j;
	printf("\n");
	for(i = 0; i < Matrix::getNumRows(); i++){
		for(j = 0; j < Matrix::getNumCols(); j++){
			printf("\t%e", Matrix::get(i, j));
		}
		printf("\n");
	}
	printf("\n");
}

void Matrix::matlabSave(char *varName, char *fileName) const{
	FILE *f = fopen(fileName, "w");
	lapack_int i, j;
	if (f == NULL){
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(f,"%s = [...\n", varName);
	for(i = 0; i < Matrix::getNumRows(); i++){
		for(j = 0; j < Matrix::getNumCols(); j++){
			fprintf(f," %e ", Matrix::get(i, j));
		}
		if(i == Matrix::getNumRows() - 1){
			fprintf(f,"];");
		}else{
			fprintf(f,";...\n");
		}
	}
	fclose(f);
}

void Matrix::matlabLoad(char *fileName){
	FILE *f = fopen(fileName, "r");
	char readBuffer[100];
	char *tokenBuffer;
	char *pEnd = NULL;
	char c;
	lapack_int i, j, k, numCols, numRows;
	if (f == NULL){
		printf("Error opening file!\n");
		exit(1);
	}
	//skip first row by iterating until first new line:
	c = getc(f);
	while((c != '\n') && (c != EOF)){
		c = getc(f);
	}
	//Now count the number of entries in the first row of the matrix
	//First we need to load the read buffer
	c = getc(f);
	i = 0;
	while((c != '\n') && (c != EOF)){
		readBuffer[i] = c;
		c = getc(f);
		i++;
	}
	//Add null char to end.
	readBuffer[i] = '\0';
	//now tokenize the read in buffer, and count the tokens.
	//that is the number of columns of the matrix.
	tokenBuffer = strtok(readBuffer, " ],;");
	numCols = 0;
	while (tokenBuffer != NULL){
		if(strcmp(tokenBuffer, (char*)"...") != 0){
			numCols++;
		}
		tokenBuffer = strtok (NULL, " ],;");
	}
	//Now count the number of newlines. That's the number of rows
	rewind(f);
	numRows = -2;
	c = getc(f);
	while(c != EOF){
		c = getc(f);
		if(c == '\n' || c == EOF){
			numRows++;
		}
	}
	//Now we know how much memory to allocate. Load the matrix into A.
	*this = Matrix(numRows, numCols, '0');
	i = 0;
	k = 0;
	rewind(f);
	//skip first row by iterating until first new line:
	c = getc(f);
	while((c != '\n') && (c != EOF)){
		c = getc(f);
	}
	c = getc(f);
	while(k < numRows){
		c = getc(f);
		i = 0;
		j = 0;
		while((c != '\n') && (c != EOF)){
			readBuffer[i] = c;
			c = getc(f);
			i++;
		}
		//Add null char to end.
		readBuffer[i] = '\0';
		//now tokenize the read in buffer, and count the tokens.
		//that is the number of columns of the matrix.
		tokenBuffer = strtok(readBuffer, " ],;");
		numCols = 0;
		while (tokenBuffer != NULL){
			if( strcmp(tokenBuffer, (char*)"...") != 0 ){
				this->entry(k,j) = strtod(tokenBuffer, &pEnd);
				j++;
			}
			tokenBuffer = strtok (NULL, " ],;");
		}
		k++;
	}
	fclose(f);
}

//end namespace moctalab
