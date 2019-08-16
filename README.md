# htucker-mpi Fokker-Planck 6 Dimensional Solver
An application of the htucker-mpi library to solving an advection-diffusion problem.

# Dependencies:
- LAPACK, the fortran linear algebra library
- LAPACKE, the C interface to to LAPACK
- CBLAS, the basic linear algebra subroutines for C
- OpenMP, shared memory parallelism
- MPI, distributed memory parallelism
- C++ Compiler

# Deemo Usage
The most portable way to use this code is to ensure that all dependencies satisfied, and then compile. After that, there is will be an executable named 
    
    fp6dDemo.x

in the same directory as the makefile. To run the demo, use the command

    mpirun -np 11 ./fp6dDemo.x  [gridSize] [maxRank] [numIteration] [finalTime]

in your terminal. Please note that a total of 11 MPI threads are used, so it is best to ensure that you have at least 11 logical processors on your computer.

Notes on command line arguments:

    gridSize:     The number of interior points on the side of a high dimensional box.
    maxRank:      The maximum rank the HTucker truncaiton algorithm truncates to.
    numIteration: The number of time steps.
    finalTime:    The end time of the PDE solver.



# Compiling
Linking is handled automatically assuming that you have the dependencies suggested by the master branch readme. Just run make in the same directory as the makefile.
