/*
 * C Source file Template by Bram Rodgers.
 * Original Draft Dated: 25, Feb 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "BoundaryCondition.h"
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
Matrix linspace(double leftVal, double rightVal, lapack_int numPoints){
	Matrix x = Matrix(numPoints, 1);
	lapack_int k = 0;
	while(k < numPoints){
		x(k) = leftVal + ((k*(rightVal - leftVal))/(numPoints-1));
		k++;
	}
	return x;
}

Matrix makeGrid(double leftVal, double rightVal, lapack_int N){
	Matrix x = linspace(leftVal, rightVal, N+2);
	Matrix y = Matrix(x.getNumRows()-1, 1);
	lapack_int i;
	for(i = 0; i < y.getNumComponents(); i++){
		y(i) = x(i);
	}
	return y;
}

Matrix fourierSineDiff(double leftVal, double rightVal, lapack_int Np1){
	double pi = acos(-1);
	lapack_int p, j;
	Matrix eta = linspace(0, 2*pi, Np1+1);
	Matrix D_spctr = Matrix(Np1, Np1);
	if(Np1 % 2 == 0){
		printf("\nFourier Collocaiton:\n"
				"Number of Interior Points must be even.\n");
		exit(1);
	}
	for(p = 0; p < Np1; p++){
		for(j = 0; j < Np1; j++){
			if(p != j){
				D_spctr(p,j) = (0.5/sin((eta(p) - eta(j))/2));
				if(((p+j)%2) == 1){
					D_spctr(p,j) = -D_spctr(p,j);
				}
			}else{
				D_spctr(p,j) = 0;
			}
		}
	}
	D_spctr = D_spctr*((2*pi)/(rightVal-leftVal));
	return D_spctr;
}

Matrix d2FiniteDiffDirich(lapack_int numPoints, double dx){
	Matrix D2_fd_mat = Matrix(numPoints, numPoints, '0');
	int i = 0;
	if(numPoints >= 3){
		for(i = 0; i < numPoints; i++){
			if(i == 0){
				D2_fd_mat(i, i) = -2/(dx*dx);
				D2_fd_mat(i, i+1) = 1/(dx*dx);
			}else if(i == numPoints-1){
				D2_fd_mat(i, i) = -2/(dx*dx);
				D2_fd_mat(i, i-1) = 1/(dx*dx);
			}else{
				D2_fd_mat(i, i-1) = 1/(dx*dx);
				D2_fd_mat(i, i) = -2/(dx*dx);
				D2_fd_mat(i, i+1) = 1/(dx*dx);
			}
		}
	}else{
		printf("\nMinimal Sampling of 3 points for 2nd order Finite Diff.\n");
	}
	return D2_fd_mat;
}

Matrix d2FiniteDiff_O4(lapack_int numPoints, double dx){
	Matrix D2_fd_mat = Matrix(numPoints, numPoints, '0');
	int i = 0;
	if(numPoints >= 7){
		for(i = 0; i < numPoints; i++){
			if(i == 0 || i == 1){
				D2_fd_mat(i, i) = 15/(4*dx*dx);
				D2_fd_mat(i, 1+i) = -77/(6*dx*dx);
				D2_fd_mat(i, 2+i) = 107/(6*dx*dx);
				D2_fd_mat(i, 3+i) = -13/(dx*dx);
				D2_fd_mat(i, 4+i) = 61/(12*dx*dx);
				D2_fd_mat(i, 5+i) = -5/(6*dx*dx);
			}else if(i == numPoints-1 || i == numPoints-2){
				D2_fd_mat(i, i) = 15/(4*dx*dx);
				D2_fd_mat(i, i-1) = -77/(6*dx*dx);
				D2_fd_mat(i, i-2) = 107/(6*dx*dx);
				D2_fd_mat(i, i-3) = -13/(dx*dx);
				D2_fd_mat(i, i-4) = 61/(12*dx*dx);
				D2_fd_mat(i, i-5) = -5/(6*dx*dx);
			}else{
				D2_fd_mat(i, i-2) = -1/(12*dx*dx);				
				D2_fd_mat(i, i-1) = 4/(3*dx*dx);
				D2_fd_mat(i, i) = -5/(2*dx*dx);				
				D2_fd_mat(i, i+1) = 4/(3*dx*dx);
				D2_fd_mat(i, i+2) = -1/(12*dx*dx);
			}
		}
	}else{
		printf("\nMinimal Sampling of 7 points for 4th order Finite Diff.\n");
	}
	return D2_fd_mat;
}
