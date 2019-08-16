/*
 * Approximate Solution to a 6d Fokker-Planck Equation by Bram Rodgers.
 * Original Draft Dated: 19, Jul 2019
 */

/*
 * Macros and Includes go here: (Some common ones listed)
 */
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include"../HTuckerMPI/HTuckerMPI.h"
#include"BoundaryCondition/BoundaryCondition.h"
#include"../HTuckerMPI/Tensor/Matrix/Matrix.h"
#ifndef NULL
	#define NULL 0
#endif


/*
 * A global variable handling the first derivative matrix
 */
Matrix DiffMat_1;
/*
 * A global variable handling the second derivative matrix
 */
Matrix DiffMat_2;
//max rank on tree
lapack_int maxRank;

/*
 * Computes a 2nd order FD integral approximation
 */
double integrateHT(HTuckerMPI& f, double dx);
/*
 * Computes a 2nd order FD integral approximation of k^th marginal pdf
 */
Matrix getMarginal(HTuckerMPI& f, double dx, lapack_int k);
/*
 * A global variable for storing grid elements along one index of the box
 */
Matrix gridVector;
/*
 * RK2 time Step function used for generating the first iteration given the
 * initial condition
 */
HTuckerMPI rk2( HTuckerMPI (*vField)(HTuckerMPI&),
				HTuckerMPI& xOld,
				double dt);
/*
 * Adams-Bashforth 2-step method for solve an ODE. needs 2 iterates to generate
 * next step in ODE
 */
HTuckerMPI ab2(HTuckerMPI (*vField)(HTuckerMPI&),
				HTuckerMPI& xOld, HTuckerMPI& xOldOld, double dt);
HTuckerMPI EulerForwardStep(HTuckerMPI (*vecField)(HTuckerMPI&),
							double dt,
							HTuckerMPI& x);
/*
 * Function declarations go here:
 */
//Declaration of a Fokker-Planck spacial operator in 6 dimensions
HTuckerMPI fp6d(HTuckerMPI& x);
//Declaration of the spacially dependent drift operators
HTuckerMPI drift1(HTuckerMPI& x);
HTuckerMPI drift2(HTuckerMPI& x);
HTuckerMPI drift3(HTuckerMPI& x);
HTuckerMPI drift4(HTuckerMPI& x);
HTuckerMPI drift5(HTuckerMPI& x);
HTuckerMPI drift6(HTuckerMPI& x);
//Declaration of the spacially dependent diffusion operators.
//Diffusion matrix is symmetric, so we only need to declare the upper triangle.
//Row 1 of diffusion matrix
HTuckerMPI diffuse11(HTuckerMPI& x);
HTuckerMPI diffuse12(HTuckerMPI& x);
HTuckerMPI diffuse13(HTuckerMPI& x);
HTuckerMPI diffuse14(HTuckerMPI& x);
HTuckerMPI diffuse15(HTuckerMPI& x);
HTuckerMPI diffuse16(HTuckerMPI& x);
//Row 2 of diffusion matrix
HTuckerMPI diffuse22(HTuckerMPI& x);
HTuckerMPI diffuse23(HTuckerMPI& x);
HTuckerMPI diffuse24(HTuckerMPI& x);
HTuckerMPI diffuse25(HTuckerMPI& x);
HTuckerMPI diffuse26(HTuckerMPI& x);
//Row 3 of diffusion matrix
HTuckerMPI diffuse33(HTuckerMPI& x);
HTuckerMPI diffuse34(HTuckerMPI& x);
HTuckerMPI diffuse35(HTuckerMPI& x);
HTuckerMPI diffuse36(HTuckerMPI& x);
//Row 4 of diffusion matrix
HTuckerMPI diffuse44(HTuckerMPI& x);
HTuckerMPI diffuse45(HTuckerMPI& x);
HTuckerMPI diffuse46(HTuckerMPI& x);
//Row 5 of diffusion matrix
HTuckerMPI diffuse55(HTuckerMPI& x);
HTuckerMPI diffuse56(HTuckerMPI& x);
//Row 6 of diffusion matrix
HTuckerMPI diffuse66(HTuckerMPI& x);
//A set of scalar functions for the drift operators
double driftCoef1(double a);
double driftCoef2(double a);
double driftCoef3(double a);
double driftCoef4(double a);
double driftCoef5(double a);
double driftCoef6(double a);
//A set of scalar functions for all the diffusion operators
//Row 1 of diffusion matrix
double diffuseCoef11(double a);
double diffuseCoef12(double a);
double diffuseCoef13(double a);
double diffuseCoef14(double a);
double diffuseCoef15(double a);
double diffuseCoef16(double a);
//Row 2 of diffusion matrix
double diffuseCoef22(double a);
double diffuseCoef23(double a);
double diffuseCoef24(double a);
double diffuseCoef25(double a);
double diffuseCoef26(double a);
//Row 3 of diffusion matrix
double diffuseCoef33(double a);
double diffuseCoef34(double a);
double diffuseCoef35(double a);
double diffuseCoef36(double a);
//Row 4 of diffusion matrix
double diffuseCoef44(double a);
double diffuseCoef45(double a);
double diffuseCoef46(double a);
//Row 5 of diffusion matrix
double diffuseCoef55(double a);
double diffuseCoef56(double a);
//Row 6 of diffusion matrix
double diffuseCoef66(double a);
/*
 * Main function for running adams-bashforth code
 */
int main(int argc, char** argv){
	// Variable Declarations:
	MPI_Init(&argc, &argv);
	char* pEnd = NULL; //Points to end of a parsed string
	char *cmndBuffer; //buffer for printing data to files
	char *fileName;//name of file to store marginal PDFs in
	char *varName;//name of variable to store marginal pdfs in
	int taskID, sysStat;
	lapack_int i, dim, gridSize, numIter, tensorSideLength, ioIndex;
	extern lapack_int  maxRank;
	double finalTime, leftVal, rightVal, dt;
	double areaUnderCurve =  0.0;
	double norm2_ht_trunc = 0.0,norm2_ht_no_trunc=0.0;
	std::valarray<lapack_int> sizeVector;
	extern Matrix gridVector;//spacial grid for creating IC and diff matrices
	extern Matrix DiffMat_1;//first order derivative matrix
	extern Matrix DiffMat_2;//second order derivative matirix
	Matrix marginalPDF;
	MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
	HTuckerMPI x0;// HTucker format tensor, a parallel object for initial cond
	HTuckerMPI x_new;// HTucker format tensor, contains new time step n
	HTuckerMPI x_old;// HTucker format tensor, contains old time step n-1
	//take in dim, tensor side length, rank, num iterations, integration time
	if(argc < 5){
		if(taskID == 0){
			printf("\nError: not enough CLI arguments.\n"
					"\n Usage:\n"
					"%s [gridSize] [maxRank] [numIter] [finalTime]\n",
					argv[0]);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		exit(1);
	}
	dim = 6;//6 dimensional fokker-planck
	gridSize = 2*(((lapack_int) strtol(argv[1], &pEnd, 0))/2);
	maxRank = (lapack_int) strtol(argv[2], &pEnd, 0);
	numIter = (lapack_int) strtol(argv[3], &pEnd, 0);
	finalTime = strtod(argv[4], &pEnd);
	//set periodic box on zero to 2*pi
	leftVal = 0.0;
	rightVal = 2.0*acos(-1.0);
	//set time step
	dt = finalTime/numIter;
	//create spacial grid
	gridVector = makeGrid(leftVal, rightVal, gridSize);
	tensorSideLength = gridVector.getNumComponents();
	//create spacial differentiation matrix
	DiffMat_1 = fourierSineDiff(leftVal, rightVal, gridVector.getNumComponents());
	//create second derivative matrix
	DiffMat_2 = DiffMat_1 * DiffMat_1;
	//initialize the HTucker tensor initial condition object
	sizeVector.resize(dim);
	sizeVector = tensorSideLength;
	x0 = HTuckerMPI(sizeVector, dim);
	//set initial condition to be a separable function

	if(x0.getThreadID() == 0){
		x0.n.D = Matrix(1,1,'1');
	}else if(x0.getTreeIndexSize() == 1){
		x0.n.D = Matrix(tensorSideLength, 1);
		for(i = 0; i < tensorSideLength; i++){
			x0.n.D.components[i] = sin(gridVector(i))*sin(gridVector(i))/acos(-1.0);
		}
	}else{
		sizeVector.resize(3);
		sizeVector = 1;
		x0.n.D = Tensor(sizeVector, 3);
		x0.n.D.components[0] = 1;
	}

	norm2_ht_no_trunc = x0.norm2();
	norm2_ht_trunc = norm2_ht_no_trunc;

	areaUnderCurve = integrateHT(x0,gridVector(1) - gridVector(0));
	if(x0.getThreadID() == 0){//otherwise, record the norm
		cmndBuffer  = (char*) malloc(sizeof(char)*200);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_trunc_norm2.dat",
							norm2_ht_trunc , x0.getThreadID());
		sysStat = system(cmndBuffer);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_no_trunc_norm2.dat",
							norm2_ht_no_trunc , x0.getThreadID());
		sysStat = system(cmndBuffer);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_integral.dat",
							areaUnderCurve , x0.getThreadID());
		sysStat = system(cmndBuffer);
		free(cmndBuffer);
	}else{
		cmndBuffer  = (char*) malloc(sizeof(char)*200);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%d\" | cat >> node_%d_rank.dat",
							1, x0.getThreadID());
		sysStat = system(cmndBuffer);
		free(cmndBuffer);
	}
	//
	//iteration one, set x_old to initial state
	x_old=x0;
	//do one step of euler forward;
	//x_new = fp6d(x_old);
	x_new = EulerForwardStep(&fp6d, dt, x_old);
	printf("\ntask_id=%d,\nmaxRank=%ld\n",taskID,maxRank);
	norm2_ht_no_trunc = x_new.norm2();
	x_new.truncateHT(maxRank);//this is solution at time i=1
	areaUnderCurve = integrateHT(x_new,gridVector(1) - gridVector(0));
	norm2_ht_trunc = x_new.norm2();
	if(x_new.getThreadID() == 0){//otherwise, record the norm
		cmndBuffer  = (char*) malloc(sizeof(char)*200);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_trunc_norm2.dat",
							norm2_ht_trunc , x_new.getThreadID());
		sysStat = system(cmndBuffer);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_no_trunc_norm2.dat",
							norm2_ht_no_trunc , x_new.getThreadID());
		sysStat = system(cmndBuffer);
		sprintf(cmndBuffer,
				"printf \"%%s\n\" \"%le\" | cat >> node_%d_integral.dat",
							areaUnderCurve , x_new.getThreadID());
		sysStat = system(cmndBuffer);
		free(cmndBuffer);
	}
	//start iterating in explicit AB2 scheme
	ioIndex = 2;
	for(i=2; i<=numIter; i++){
		x0 = x_new;//save previous time step
		x_new = ab2(&fp6d, x0, x_old, dt);//update x_new to time i
		norm2_ht_no_trunc = x_new.norm2();
		x_new.truncateHT(maxRank);//apply truncation algorithm
		//get new value of integral
		areaUnderCurve = integrateHT(x_new,gridVector(1) - gridVector(0));
		norm2_ht_trunc = x_new.norm2();
		if(x_new.getThreadID() == 0){//otherwise, record the norm
			cmndBuffer  = (char*) malloc(sizeof(char)*200);
			sprintf(cmndBuffer,
					"printf \"%%s\n\" \"%le\" | cat >> node_%d_trunc_norm2.dat",
								norm2_ht_trunc , x_new.getThreadID());
			sysStat = system(cmndBuffer);
			sprintf(cmndBuffer,
					"printf \"%%s\n\" \"%le\" | cat >> node_%d_no_trunc_norm2.dat",
								norm2_ht_no_trunc , x_new.getThreadID());
			sysStat = system(cmndBuffer);
			sprintf(cmndBuffer,
					"printf \"%%s\n\" \"%le\" | cat >> node_%d_integral.dat",
								areaUnderCurve , x_new.getThreadID());
			sysStat = system(cmndBuffer);
			free(cmndBuffer);
		}
		//now compute x_0 direction marginal PDF.
		if(ioIndex == numIter / 5){
			ioIndex = 0;
			marginalPDF = getMarginal(x_new,gridVector(1) - gridVector(0),0);
			if(x_new.getThreadID() == 0){
				cmndBuffer  = (char*) malloc(sizeof(char)*200);
				fileName  = (char*) malloc(sizeof(char)*200);
				varName = (char*) malloc(sizeof(char)*200);
				sprintf(fileName,"marginals/axis_0_iteration_%ld.mat",i);
				sprintf(varName,"p_%ld",i);
				sprintf(cmndBuffer,"mkdir marginals");
				sysStat = system(cmndBuffer);
				sprintf(cmndBuffer,"touch marginals/axis_0_iteration_%ld.mat",i);
				marginalPDF.matlabSave(varName, fileName);
				free(cmndBuffer);
				free(varName);
				free(fileName);
			}
		}
		x_old = x0;//update previous iterate the the now old time i-1
		ioIndex++;
	}
	printf("\n%s\n\n", argv[0]);
	MPI_Finalize();
	return 0;
}

//Implementation of integrateHT method
double integrateHT(HTuckerMPI& f, double dx){
	HTuckerMPI y = f;
	Matrix w;
	Tensor fullTen;
	double integralValue = 0.0;
	if(y.getTreeIndexSize()  == 1){//if this is a leaf
		//initialize the integration weight vector
		w = Matrix(1, y.n.D.getIndexLength(0),'1');
		w = w*dx;
		w(0) = w(0)*0.5;
		w(w.getNumCols()-1) = w(w.getNumCols()-1)*0.5;
		y.n.D = w.modalProd(0,y.n.D);
	}
	y.indexLength = 1;
	fullTen = y.full();
	if(y.getThreadID() == 0){
		integralValue = fullTen.components[0];
	}
	return integralValue;
}

//Implementation of get marginal PDF method.
Matrix getMarginal(HTuckerMPI& f, double dx, lapack_int k){
	HTuckerMPI y = f;
	Matrix w, M;
	Tensor fullTen;
	double integralValue = 0.0;
	//if this is a leaf and is not the k^th variable index
	if(y.getTreeIndexSize()  == 1 && k != y.getTreeIndex(0)){
		//initialize the integration weight vector
		w = Matrix(1, y.n.D.getIndexLength(0),'1');
		w = w*dx;
		w(0) = w(0)*0.5;
		w(w.getNumCols()-1) = w(w.getNumCols()-1)*0.5;
		y.n.D = w.modalProd(0,y.n.D);
	}
	M = y.contractedVector(k);
	return M;
}

//Implementation of rk2 method
HTuckerMPI rk2( HTuckerMPI (*vField)(HTuckerMPI&),
				HTuckerMPI& xOld,
				double dt){
	HTuckerMPI k = vField(xOld);
	if(k.getThreadID() == 0){
		k.scale(0.5*dt);
	}
	k = xOld + k;
	k = vField(k);
	if(k.getThreadID() == 0){
		k.scale(dt);
	}
	k = xOld + k;
	return k;
}
//Implementation of ab2 method
HTuckerMPI ab2(HTuckerMPI (*vField)(HTuckerMPI&),
				HTuckerMPI& xOld, HTuckerMPI& xOldOld, double dt){
	HTuckerMPI k1 = vField(xOld);
	HTuckerMPI k2 = vField(xOldOld);
	if(k1.getThreadID() == 0){
		k1.scale(1.5*dt);
	}
	if(k2.getThreadID() == 0){
		k2.scale(-0.5*dt);
	}
	k1 = k1+xOld;
	k1 = k1+k2;
	return k1;
}

//Implementation of Fokker-Plank 6d operator
HTuckerMPI fp6d(HTuckerMPI& x){
	HTuckerMPI y = drift1(x);
	y = y+drift2(x);
	y = y+drift3(x);
	y = y+drift4(x);
	y = y+drift5(x);
	y = y+drift6(x);

	y = y+diffuse11(x);
	y = y+diffuse12(x);
	y = y+diffuse13(x);
	y = y+diffuse14(x);
	y = y+diffuse15(x);
	y = y+diffuse16(x);
	y = y+diffuse22(x);
	y = y+diffuse23(x);
	y = y+diffuse24(x);
	y = y+diffuse25(x);
	y = y+diffuse26(x);
	y = y+diffuse33(x);
	y = y+diffuse34(x);
	y = y+diffuse35(x);
	y = y+diffuse36(x);
	y = y+diffuse44(x);
	y = y+diffuse45(x);
	y = y+diffuse46(x);
	y = y+diffuse55(x);
	y = y+diffuse56(x);
	y = y+diffuse66(x);
	return y;
}

//Implementations for 6 drift operators
HTuckerMPI drift1(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 0
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 1
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef1(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI drift2(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 1
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 2){//if this is leaf 2
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef2(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI drift3(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 2){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 3){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef3(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI drift4(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 3){//if this is leaf 3
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 4
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef4(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI drift5(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 4){//if this is leaf 4
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef5(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI drift6(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 5){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 0){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, driftCoef6(gridVector(k)));
			}
		}
	}
	return y;
}


//Implementation of the spacially dependent diffusion operators.
//Diffusion matrix is symmetric, so we only need to declare the upper triangle.
//Row 1 of diffusion matrix
HTuckerMPI diffuse11(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef11(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI diffuse12(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef12(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse13(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef13(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse14(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef14(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse15(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 3){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef15(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse16(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef16(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
//Row 2 of diffusion matrix
HTuckerMPI diffuse22(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 2){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef22(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI diffuse23(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef23(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 3
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse24(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef24(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse25(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef25(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse26(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef26(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
//Row 3 of diffusion matrix
HTuckerMPI diffuse33(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef33(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI diffuse34(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef34(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse35(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef35(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse36(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef36(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
//Row 4 of diffusion matrix
HTuckerMPI diffuse44(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 2){//if this is leaf 3
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef44(gridVector(k)));
			}
		}
	}
	return y;
}
HTuckerMPI diffuse45(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 0){//if this is leaf 1
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef45(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
HTuckerMPI diffuse46(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 3){//if this is leaf 4
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef46(gridVector(k)));
			}
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
//Row 5 of diffusion matrix
HTuckerMPI diffuse55(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef55(gridVector(k)));
			}
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
	}
	return y;
}
HTuckerMPI diffuse56(HTuckerMPI& x){
	extern Matrix DiffMat_1;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 4){//if this is leaf 5
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
		if(y.getTreeIndex(0) == 1){//if this is leaf 2
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef56(gridVector(k)));
			}
		}
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			y.leafMult(DiffMat_1);//apply differentiation operator
		}
	}else if(y.getThreadID() == 0){//if this is root node
		y.scale(2.0);
	}
	return y;
}
//Row 6 of diffusion matrix
HTuckerMPI diffuse66(HTuckerMPI& x){
	extern Matrix DiffMat_2;
	extern Matrix gridVector;
	HTuckerMPI y = x;
	lapack_int k;
	if(y.getTreeIndexSize() == 1){//if this is a leaf
		if(y.getTreeIndex(0) == 5){//if this is leaf 6
			for(k = 0; k < gridVector.getNumComponents(); k++){
				((Matrix*)&(y.n.D))->scaleRow(k, diffuseCoef66(gridVector(k)));
			}
			y.leafMult(DiffMat_2);//apply differentiation operator
		}
	}
	return y;
}

//Implementations of scalar functions for the drift operators
double driftCoef1(double a){return cos(a);}
double driftCoef2(double a){return sin(a);}
double driftCoef3(double a){return cos(2.0*a);}
double driftCoef4(double a){return sin(2.0*a);}
double driftCoef5(double a){return cos(3.0*a);}
double driftCoef6(double a){return sin(3.0*a);}
//Implementations of scalar functions for all the diffusion operators
//Row 1 of diffusion matrix
double diffuseCoef11(double a){return (5.0*pow((cos(a)),2)+6.0);}
double diffuseCoef12(double a){return pow(sin(a),2);}
double diffuseCoef13(double a){return pow(cos(a),2);}
double diffuseCoef14(double a){return pow(sin(a),2);}
double diffuseCoef15(double a){return pow(cos(a),2);}
double diffuseCoef16(double a){return pow(sin(a),2);}
//Row 2 of diffusion matrix
double diffuseCoef22(double a){return (5.0*pow((cos(3.0*a)),2)+6.0);}
double diffuseCoef23(double a){return sin(5.0*a);}
double diffuseCoef24(double a){return cos(2.0*a);}
double diffuseCoef25(double a){return sin(4.0*a);}
double diffuseCoef26(double a){return cos(a);}
//Row 3 of diffusion matrix
double diffuseCoef33(double a){return(5.0*pow((cos(3.0*a)),2)+6.0);}
double diffuseCoef34(double a){return sin(2.0*a);}
double diffuseCoef35(double a){return cos(6.0*a);}
double diffuseCoef36(double a){return sin(a);}
//Row 4 of diffusion matrix
double diffuseCoef44(double a){return (5.0*pow((cos(a)),2)+6.0);}
double diffuseCoef45(double a){return sin(a);}
double diffuseCoef46(double a){return cos(4.0*a);}
//Row 5 of diffusion matrix
double diffuseCoef55(double a){return (5.0*pow((cos(5.0*a)),2)+6.0);}
double diffuseCoef56(double a){return sin(a);}
//Row 6 of diffusion matrix
double diffuseCoef66(double a){return (5.0*pow(cos(7.0*a),2) +6.0);}
HTuckerMPI EulerForwardStep(HTuckerMPI (*vecField)(HTuckerMPI&),
							double dt,
							HTuckerMPI& x){
	HTuckerMPI x_1 = vecField(x);
	if(x.getThreadID() == 0){//Do Scaling update.
		x_1.scale(dt);
	}
	x_1 = x + x_1;
	return x_1;
}
