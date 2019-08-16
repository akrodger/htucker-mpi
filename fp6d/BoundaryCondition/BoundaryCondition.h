#ifndef BOUNDARYCONDITIONLIB_H
#define BOUNDARYCONDITIONLIB_H

/*
 * C Header file Template by Bram Rodgers.
 * Original Draft Dated: 25, Feb 2018
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include<cmath>
#include"../../HTuckerMPI/Tensor/Matrix/Matrix.h"
#ifndef NULL
	#define NULL 0
#endif

/*
 * Object and Struct Definitions:
 */

/*
 * Function Declarations:
 */
Matrix linspace(double leftVal, double rightVal, lapack_int numPoints);

//grid construction function for use in fourier sine collocation
Matrix makeGrid(double leftVal, double rightVal, lapack_int N);

Matrix fourierSineDiff(double leftVal, double rightVal, lapack_int numPoints);

//A 2nd order 2nd deriv finite difference matrix for use in solutions to BVPs
Matrix d2FiniteDiffDirich(lapack_int numPoints, double dx);

//4th order 2nd deriv finite matrix. Uses forward and backward differences
//near endpoints
Matrix d2FiniteDiff_O4(lapack_int numPoints, double dx);

#endif
