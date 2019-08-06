################################################################################
##                                                                            ##
##           Makefile for htucker-mpi tensor decomposition library            ##
##                       Written by Bram Rodgers                              ##
##                Initial draft written September 5, 2018                     ##
##                                                                            ##
################################################################################

##  Dependencies:
##  LAPACK, the fortran linear algebra library
##  LAPACKE, the C interface to to LAPACK
##  CBLAS, the basic linear algebra subroutines for C
##	OpenMP, shared memory parallelism
##	MPI, distributed memory parallelism
##  C++ Compiler

##  Programmed and Tested on Debian 9 Stable and .
##  Defaults package names for dependencies:
##  LAPACK        :  liblapack3
##  LAPACKE       :  liblapacke-dev, liblapacke
##  BLAS          :  libblas-dev, libblas3
##  OpenMP        :  libgomp1
##	MPI           :  openmpi-bin
##	C++ Compiler  :  g++

##  Compiler flags for debugging
CFLAGS=-O3#-g -Wall
##  The aliases for linking the dependences. Change these to use different
##  dependency implementations
OMP=-fopenmp
LAPACK=-llapack
LAPACKE=-llapacke
BLAS=-lblas
##  The aliases for the compiler being used
MPICXX=mpic++
CXX=g++
##  These aliases are for the source locations
TENSOR_LOC=HTuckerMPI/Tensor
MATRIX_LOC=$(TENSOR_LOC)/Matrix
HTUCKER_LOC=HTuckerMPI
DISTREE_LOC=$(HTUCKER_LOC)/DisTree


HT_DEPS=$(TENSOR_LOC)/Tensor.cpp\
 $(MATRIX_LOC)/Matrix.cpp\
 $(HTUCKER_LOC)/HTuckerMPI.cpp\
 $(DISTREE_LOC)/DisTree.cpp $(DISTREE_LOC)/DisTreeNode.cpp


default: htucker-mpi.a


htucker-mpi.a: $(HT_DEPS) Tensor.o Matrix.o HTuckerMPI.o DisTreeNode.o DisTree.o
	ar rcs htucker-mpi.a\
	 Tensor.o\
	 Matrix.o\
	 HTuckerMPI.o\
	 DisTree.o DisTreeNode.o

Tensor.o:
	$(MPICXX) -c $(CFLAGS) $(OMP)\
	 $(TENSOR_LOC)/Tensor.cpp -o Tensor.o

Matrix.o: Tensor.o
	$(MPICXX) -c -fPIC $(CFLAGS) $(OMP)\
	 $(MATRIX_LOC)/Matrix.cpp -o Matrix.o

HTuckerMPI.o: DisTreeNode.o DisTree.o
	$(MPICXX) -c -fPIC $(CFLAGS) $(OMP)\
	 $(HTUCKER_LOC)/HTuckerMPI.cpp -o HTuckerMPI.o

DisTreeNode.o:
	$(MPICXX) -c -fPIC $(CFLAGS) $(OMP)\
	 $(DISTREE_LOC)/DisTreeNode.cpp -o DisTreeNode.o

DisTree.o:
	$(MPICXX) -c -fPIC $(CFLAGS) $(OMP)\
	 $(DISTREE_LOC)/DisTree.cpp -o DisTree.o

clean:
	rm *.o *.a
