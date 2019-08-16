# htucker-mpi
A high performance computing oriented parallel implementation of the Hierarchical Tucker tensor decomposition toolbox. A closely related MATLAB version can be found at https://anchp.epfl.ch/index-html/software/htucker/ .

# Dependencies:
- LAPACK, the fortran linear algebra library
- LAPACKE, the C interface to to LAPACK
- CBLAS, the basic linear algebra subroutines for C
- OpenMP, shared memory parallelism
- MPI, distributed memory parallelism
- C++ Compiler

# Library Usage
The most portable way to use this code is to ensure that all dependencies satisfied, and then compile. After that, simply place the file htucker-mpi.a to a location where you keep accessible linked libraries. Linking all dependencies is then accomplished by appending
    
    -llapack -llapacke -lblas -lm -l/[location of libs for linking]/htucker-mpi.a

to the linking section of the C++ compilation and linking commands.

# Compiling
Make sure all your dependencies are installed. The ones used by default in this library are listed with the following names in the Ubuntu 19.04 repositories.

-libgomp1
-liblapack3
-liblapacke
-libblas3
-libgslcblas0
-openmpi-bin
-g++

If these are installed, then compilation of the linkable library htucker-mpi.a should easily completed by running make in the same directory as this README file.
