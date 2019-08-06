/*
 * C Source file Template by Bram Rodgers.
 * Original Draft Dated: 25, Feb 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "HTuckerMPI.h"
#include "stdio.h"
#include "stdlib.h"
#ifndef NULL
	#define NULL 0
#endif

/*
 * Locally used helper functions:
 */

/*
 * Static Local Variables:
 */

HTuckerMPI::HTuckerMPI(){
	this->n = DisTreeNode();
	this->indexLength.resize(0);
}

HTuckerMPI::HTuckerMPI(std::valarray<lapack_int>& sizeVector,
						lapack_int tensorDim){
	//MPI task handling variables
	int taskID, numTasks, tag=1, i, j;
	std::valarray<int> sendIntBuff;
	std::valarray<int> recvIntBuff;
	std::valarray<int> sendIndexBuff;
	std::valarray<int> recvIndexBuff;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	MPI_Status stat;
	//Synchronize the tasks before starting construction
	MPI_Barrier(MPI_COMM_WORLD);
	//Get the information on how many tasks there are and what this task's
	//ID number is
	MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	//Set the node id to the retrieved task ID
	this->n.t_id = taskID;
	this->indexLength.resize(tensorDim);
	for(i = 0; i < tensorDim; i++){
		this->indexLength[i] = sizeVector[i];
	}
	//If this is the root, then we need to compute the tree structure
	if(taskID == 0){
		//Create a dimension tree based on the number of indexes in T
		DisTree t = DisTree((int) tensorDim);
		//If we don't have exactly the right amount of cores, crash
		if(numTasks != (int)t.tree.size()){
			printf("\nERROR: INCORRECT NUMBER OF MPI TASKS.\n\n"
					"Required number of tasks:\t%d\n"
					"Provided number of tasks:\t%d\n",
					(int)t.tree.size(), numTasks);
			
			exit(1);
		}
		this->n = t.tree[0]; //Set root node to root of computed dimension tree
		sendIntBuff.resize(4);//resize buff to send parent, left, and right IDs
		//The sendIntBuff[3] entry contains the length of the node index
		//Fill the long buffer with the dimension.
		sendLongBuff.resize(1);
		//0 index: dim of tensor
		sendLongBuff[0] = (long) tensorDim;
		//Start a loop to load tree node IDs into distributed cores/tasks
		for(i = 1; i < numTasks; i++){//Start at the first non-root node
			sendIntBuff[0] = t.tree[i].p_id;
			sendIntBuff[1] = t.tree[i].l_id;
			sendIntBuff[2] = t.tree[i].r_id;
			sendIntBuff[3] = (int) t.tree[i].index.size();
			//Now fill the index buffer with this tree node's index
			sendIndexBuff.resize(sendIntBuff[3]);
			for(j = 0; j < sendIntBuff[3]; j++){
				sendIndexBuff[j] = t.tree[i].index[j];
			}
			//Send all the parent, left, and right IDs
			MPI_Send(&(sendIntBuff[0]), 4, MPI_INT, i, tag, MPI_COMM_WORLD);
			//Send a message saying what the dim of tensor is
			MPI_Send(	&(sendLongBuff[0]),
						1, MPI_LONG, i, tag, MPI_COMM_WORLD);
			//Send a message containing the index of this tensor tree node
			MPI_Send(	&(sendIndexBuff[0]), sendIntBuff[3], MPI_INT, i, 
							tag, MPI_COMM_WORLD);
		}
	}else{//Otherwise wait to recieve this task's node data
		//resize the recieve buffer for the parent, left, and right ids
		//as well as the index length
		recvIntBuff.resize(4);
		//resize the long buffer to learn dimension of tensor to be made
		recvLongBuff.resize(1);
		//Recieve the message buffered by the root node
		//This message contains the info on the parent and child locations
		MPI_Recv(&(recvIntBuff[0]), 4, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);
		//Get the dim for this tensor
		MPI_Recv(	&(recvLongBuff[0]),
					1, MPI_LONG, 0, tag, MPI_COMM_WORLD, &stat);
		//Assign memory to index recieve buffer to get this tree node's index
		recvIndexBuff.resize(recvIntBuff[3]);
		//Recieve message containing  index of this tensor tree node
		MPI_Recv(&(recvIndexBuff[0]), recvIntBuff[3], MPI_INT, 0, 
					tag, MPI_COMM_WORLD, &stat);
		//Now that message passing is finished, assign the values of
		//the node object
		this->n.p_id = recvIntBuff[0];
		this->n.l_id = recvIntBuff[1];
		this->n.r_id = recvIntBuff[2];
		this->n.index.resize(recvIntBuff[3]);
		for(i = 0; i < recvIntBuff[3]; i++){
			this->n.index[i] = (lapack_int) recvIndexBuff[i];
		}
	}
}

HTuckerMPI::HTuckerMPI(Tensor& T, lapack_int maxRank){
	//MPI task handling variables
	int taskID, numTasks, tag=1, i, j;
	std::valarray<int> sendIntBuff;
	std::valarray<int> recvIntBuff;
	std::valarray<int> sendIndexBuff;
	std::valarray<int> recvIndexBuff;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	Tensor B_t;
	std::valarray<lapack_int> sizeVector;// multiIndex;
	Matrix U_t, U_l, U_r, lCol, rCol;
	MPI_Status stat;
	//Synchronize the tasks before starting construction
	MPI_Barrier(MPI_COMM_WORLD);
	//Get the information on how many tasks there are and what this task's
	//ID number is
	MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	//Set the node id to the retrieved task ID
	this->n.t_id = taskID;
	//If this is the root, then we need to compute the tree structure
	if(taskID == 0){
		//Create a dimension tree based on the number of indexes in T
		DisTree t = DisTree((int) T.getDim());
		//If we don't have exactly the right amount of cores, crash
		if(numTasks != (int)t.tree.size()){
			printf("\nERROR: INCORRECT NUMBER OF MPI TASKS.\n\n"
					"Required number of tasks:\t%d\n"
					"Provided number of tasks:\t%d\n",
					(int)t.tree.size(), numTasks);
			
			exit(1);
		}
		this->n = t.tree[0]; //Set root node to root of computed dimension tree
		sendIntBuff.resize(4);//resize buff to send parent, left, and right IDs
		//The sendIntBuff[3] entry contains the length of the node index
		sendDoubleBuff = T.components; //copy T components into send buffer
		//Fill the long buffer with the dimension depth data of T.
		sendLongBuff.resize(T.getDim() + 2);
		//0 index: number of components in T.
		sendLongBuff[0] = (long) T.getNumComponents();
		//1 index: dim of T
		sendLongBuff[1] = (long) T.getDim();
		//2 , ... , d + 1 indexes : index lengths of of T
		this->indexLength.resize(T.getDim());
		for(i = 0; i < T.getDim(); i++){
			sendLongBuff[i+2] = T.getIndexLength(i);
			this->indexLength[i] = T.getIndexLength(i);
		}
		//Start a loop to load tree node IDs into distributed cores/tasks
		for(i = 1; i < numTasks; i++){//Start at the first non-root node
			sendIntBuff[0] = t.tree[i].p_id;
			sendIntBuff[1] = t.tree[i].l_id;
			sendIntBuff[2] = t.tree[i].r_id;
			sendIntBuff[3] = (int) t.tree[i].index.size();
			//Now fill the index buffer with this tree node's index
			sendIndexBuff.resize(sendIntBuff[3]);
			for(j = 0; j < sendIntBuff[3]; j++){
				sendIndexBuff[j] = t.tree[i].index[j];
			}
			//Send all the parent, left, and right IDs
			MPI_Send(&(sendIntBuff[0]), 4, MPI_INT, i, tag, MPI_COMM_WORLD);
			//Send a message saying how long the double buffer is and 
			//also what the dim of T is
			MPI_Send(	&(sendLongBuff[0]),
						2, MPI_LONG, i, tag, MPI_COMM_WORLD);
			//Now that the reciever has allocated memory, send the rest of
			//the message containing the index depths of T
			MPI_Send(	&(sendLongBuff[2]),
						(int) T.getDim(), MPI_LONG, i, tag, MPI_COMM_WORLD);
			//Send a message containing the components of the input tensor
			MPI_Send(	&(sendDoubleBuff[0]),
						(int) sendLongBuff[0],
						MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
			//Send a message containing the index of this tensor tree node
			MPI_Send(	&(sendIndexBuff[0]), sendIntBuff[3], MPI_INT, i, 
							tag, MPI_COMM_WORLD);
		}
	}else{//Otherwise wait to recieve this task's node data
		//resize the recieve buffer for the parent, left, and right ids
		//as well as the index length
		recvIntBuff.resize(4);
		//resize the long buffer to learn how many elements are in the double
		//buffer
		recvLongBuff.resize(2);
		//Recieve the message buffered by the root node
		//This message contains the info on the parent and child locations
		MPI_Recv(&(recvIntBuff[0]), 4, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);
		//Get the number of components in T and also its dim
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, 0, tag, MPI_COMM_WORLD, &stat);
		//allocate the indicated amount of memory into the double buffer
		recvDoubleBuff.resize(recvLongBuff[0]);
		recvLongBuff.resize(recvLongBuff[1]);
		//Now that memory is allocated to the dimension buffer, recieve the
		//list of dimension lengths
		MPI_Recv(	&(recvLongBuff[0]),
					(int) recvLongBuff.size(), MPI_LONG, 0,
					tag, MPI_COMM_WORLD, &stat);
		//Recieve the components of the input tensor
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &stat);
		//Assign memory to index recieve buffer to get this tree node's index
		recvIndexBuff.resize(recvIntBuff[3]);
		//Recieve message containing  index of this tensor tree node
		MPI_Recv(&(recvIndexBuff[0]), recvIntBuff[3], MPI_INT, 0, 
					tag, MPI_COMM_WORLD, &stat);
		//Now that message passing is finished, assign the values of
		//the node object
		this->n.p_id = recvIntBuff[0];
		this->n.l_id = recvIntBuff[1];
		this->n.r_id = recvIntBuff[2];
		this->n.index.resize(recvIntBuff[3]);
		//this->n
		//move the list of dimension lengths into the sizeVector array
		//while explicitly casting the type to lapack_int
		sizeVector.resize(recvLongBuff.size());
		this->indexLength.resize(recvLongBuff.size());
		for(i = 0; i < (int) recvLongBuff.size(); i++){
			sizeVector[i] = (lapack_int) recvLongBuff[i];
			this->indexLength[i] = (lapack_int) recvLongBuff[i];
		}
		for(i = 0; i < recvIntBuff[3]; i++){
			this->n.index[i] = (lapack_int) recvIndexBuff[i];
		}
		//Use B_t Tensor object instance to hold the full tensor T
		B_t = Tensor(sizeVector, (lapack_int) sizeVector.size());

		//Assign the components of the double buffer to the components
		B_t.components = recvDoubleBuff;
	}
	sendIntBuff.resize(0);
	recvIntBuff.resize(0);
	sendIndexBuff.resize(0);
	recvIndexBuff.resize(0);
	sendLongBuff.resize(0);
	recvLongBuff.resize(0);
	sendDoubleBuff.resize(0);
	recvDoubleBuff.resize(0);
	//Synchronize tasks after initializing the tree structure
	MPI_Barrier(MPI_COMM_WORLD);
	//At this point, we have successfully constructed the tensor tree
	//distributed across all given tasks
	//What we do next is dependent on the node ID and index length
	if(taskID == 0){//case 1: If this is the root node
		//In this case, we vectorize the given tensor and do not take an SVD
		//We then wait to recieve the frames of the child nodes
		//Then compute the root node matrix by projecting onto the kronecker 
		//of the columns of those child frames.
		//Result gets saved as a matrix in this->n.D
		U_t = Matrix(T, this->n.index, (T.getDim()/2) + (T.getDim()%2));
		//Now we recieve the left child matrix.
		//First, get the rows and cols of the left child using the long buffer
		recvLongBuff.resize(2);
	/*FILL IN RECIEVE SIZE LINE HERE*/
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		//Now recieve the components of U_l using the long buffer
	/*FILL IN RECIEVE DOUBLES LINES HERE*/
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);

		//printf("\nU_l size = \n%ld, %ld", (lapack_int)recvLongBuff[0],
		//				(lapack_int)recvLongBuff[1]);
		//Now set U_l with a constructor
		U_l = Matrix(	(lapack_int)recvLongBuff[0],
						(lapack_int)recvLongBuff[1]);
		U_l.components = recvDoubleBuff;
		//Now do the same with U_r
	/*FILL IN RECIEVE SIZE LINE HERE*/
		recvLongBuff.resize(2);
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);

		//Now recieve the components of U_l using the long buffer
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
	/*FILL IN RECIEVE DOUBLES LINES HERE*/
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		//Now set U_l with a constructor
		
		U_r = Matrix(	(lapack_int)recvLongBuff[0],
						(lapack_int)recvLongBuff[1]);
		U_r.components = recvDoubleBuff;
		//Now se the root node tensor to be a matrix with U_l cols by
		//U_r cols with entries from projecting onto leaf nodes
		this->n.D = Matrix(U_l.getNumCols(), U_r.getNumCols());
		//U_l.print();
		U_l = U_l.transp();
		//U_r.print();
		U_t = U_l * U_t * U_r;
		
		//U_r = U_r.kron(U_l);
		//U_r = U_r * U_t;
		//U_t.print();
		this->n.D.components = U_t.components;
	}else if(this->n.index.size() == 1){//case 2: if this is a leaf
		//For the leaf, we matricize, take an SVD, and then send
		//the leaf frame made of left singular vectors to the parent
		//The left singular vectors get saved in this->n.D
		B_t = Matrix(B_t, this->n.index,
							(lapack_int)this->n.index.size());
		this->n.D = ((Matrix*)(&B_t))->getLeftSV(maxRank);
		//We load 2 things into the long buffer to send to the parent node
		//Number of rows in the matrix & number of cols in the matrix,
		sendLongBuff.resize(2);
		sendLongBuff[0] = (long) this->n.D.getIndexLength(0);
		sendLongBuff[1] = (long) this->n.D.getIndexLength(1);
		
		//Also we load the double buffer their the data for the left SVs
		sendDoubleBuff = this->n.D.components;
		//Now we send the size data to the parent node
	/*FILL SEND SIZE LINE IN HERE*/
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
	/*FILL SEND DOUBLES LINE IN HERE*/
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else{//The last case is if this is an interior node
		//For this case, we matricize, and take an SVD, then wait to recieve
		//the frames from the two children nodes.
		//Once recieved, we compute the transfer tensor by projecting
		//This node's frame onto the kronecker product of the child frames cols
		//We set B_t equal to that transfer tensor and save it in this->n.D
		B_t = Matrix(B_t, this->n.index,
							(lapack_int)this->n.index.size());
		U_t = ((Matrix*)&B_t)->getLeftSV(maxRank);
		//We send the frame U_t as soon as we compute it.
		//First, send the size information about the rows and columns of U_t
		sendLongBuff.resize(2);
		sendLongBuff[0] = U_t.getNumRows();
		sendLongBuff[1] = U_t.getNumCols();
	/*FILL IN SEND SIZE LINE HERE*/
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now we send the components of U_t
		sendDoubleBuff = U_t.components;
	/*FILL IN SEND SIZE LINE HERE*/
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now we recieve the left child matrix.
		//First, get the rows and cols of the left child using the long buffer
		recvLongBuff.resize(2);
	/*FILL IN RECIEVE SIZE LINE HERE*/
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		//Now recieve the components of U_l using the long buffer
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
	/*FILL IN RECIEVE DOUBLES LINES HERE*/
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		//Now set U_l with a constructor
		U_l = Matrix(	(lapack_int)recvLongBuff[0],
						(lapack_int)recvLongBuff[1]);
		U_l.components = recvDoubleBuff;
		//Now do the same with U_r
	/*FILL IN RECIEVE SIZE LINE HERE*/
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		//Now recieve the components of U_r using the long buffer
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
	/*FILL IN RECIEVE DOUBLES LINES HERE*/
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		//Now set U_l with a constructor
		U_r = Matrix(	(lapack_int)recvLongBuff[0],
						(lapack_int)recvLongBuff[1]);
		U_r.components = recvDoubleBuff;
		//We have the 3 matrices needed to compute the transfer tensor
		//Using a triple nested for loop, we begin the projection
//		multiIndex.resize(3);
		sizeVector.resize(3);
		sizeVector[0] = U_l.getNumCols();
		sizeVector[1] = U_r.getNumCols();
		sizeVector[2] = U_t.getNumCols();
		this->n.D = Tensor(sizeVector, 3);
		this->n.D.components =
				(((U_r.transp()).kron(U_l.transp())).operator*(U_t)).components;
	}
	//Synchronize the tasks after constructing objects in each task
	MPI_Barrier(MPI_COMM_WORLD);
}

double HTuckerMPI::norm2() const{
	double norm2_val = 0;
	int N = 1;
	HTuckerMPI nrm = *this;
	nrm.orthog();
	if(this->getThreadID() == 0){
		N = (int) nrm.n.D.getNumComponents();
		norm2_val =  cblas_dnrm2(N, &(nrm.n.D.components[0]), 1);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return norm2_val;
}


double HTuckerMPI::innerProd(HTuckerMPI& x){
	Matrix U_this, U_x, M_t, M_r, M_l, B_this, B_x;
	std::valarray<int> sendIntBuff;
	std::valarray<int> recvIntBuff;
	std::valarray<int> sendIndexBuff;
	std::valarray<int> recvIndexBuff;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	std::valarray<lapack_int> sizeVector;
	MPI_Status stat;
	MPI_Barrier(MPI_COMM_WORLD);
	int tag = 1;
	double innerProdVal = 0;
	if(this->n.index.size() == 1){//if this is a leaf
		U_this = Matrix(this->n.D.getIndexLength(0),
						this->n.D.getIndexLength(1));
		U_this.components = this->n.D.components;
		U_x = Matrix(x.n.D.getIndexLength(0),
						x.n.D.getIndexLength(1));
		U_this.components = this->n.D.components;
		U_x.components = x.n.D.components;
		M_t = U_this.transp()*U_x;
		//U_this.print();
		//U_x.print();
		//M_t.print();
		//Send M_t to parent
		//tell parent how much memory to accept
		sendLongBuff.resize(2);
		sendLongBuff[0] = M_t.getIndexLength(0);
		sendLongBuff[1] = M_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(M_t.getNumComponents());
		sendDoubleBuff = M_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else if(this->n.t_id != 0){//if this thread is interior to the tree
		//Convert transfer tensor to matrix
		B_this = Matrix(this->n.D.getIndexLength(0)*
						this->n.D.getIndexLength(1),
						this->n.D.getIndexLength(2));
		B_x = Matrix(x.n.D.getIndexLength(0)*x.n.D.getIndexLength(1),
						x.n.D.getIndexLength(2));
		B_this.components = this->n.D.components;
		B_x.components = x.n.D.components;
		//set of message to recieve memory buffer size
		recvLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		M_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		M_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_r.components = recvDoubleBuff;
		//Now compute this node's matrix in inner product algorithm
		M_t = B_this.transp()*(M_r.kron(M_l))*B_x;
		//M_t.print();
		//Send M_t to parent
		//tell parent how much memory to accept
		sendLongBuff.resize(2);
		sendLongBuff[0] = M_t.getIndexLength(0);
		sendLongBuff[1] = M_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(M_t.getNumComponents());
		sendDoubleBuff = M_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else{//this is the root node
		B_this = Matrix(1,
					this->n.D.getIndexLength(0)*this->n.D.getIndexLength(1)	);
		B_x = Matrix(x.n.D.getIndexLength(0)*x.n.D.getIndexLength(1),1);
		B_this.components = this->n.D.components;
		B_x.components = x.n.D.components;
		recvLongBuff.resize(2);
		//First recieve from left child
 		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		M_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		M_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_r.components = recvDoubleBuff;
		//Now compute end of inner product function
//		M_r.print();
//		M_l.print();
//		B_t.print();
		M_t = B_this * (M_r.kron(M_l))*B_x;
		innerProdVal = M_t(0,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return innerProdVal;
}

HTuckerMPI HTuckerMPI::hada(HTuckerMPI& x){
	lapack_int i, j;
	HTuckerMPI hadaTensor = *this;
	std::valarray<lapack_int> sizeVector;
	std::valarray<lapack_int> multiIndex;
	Matrix B_this, B_x, U_this, U_x, row_this, row_x, row_hada;
	if(this->getThreadID() == 0){
		B_this = this->n.D;
		B_x = x.n.D;
		hadaTensor.n.D = B_this.kron(B_x); 
	}else if(this->getTreeIndexSize() != 1){
		sizeVector.resize(3);
		sizeVector[0] = this->n.D.getIndexLength(0)*
							x.n.D.getIndexLength(0);
		sizeVector[1] = this->n.D.getIndexLength(1)*
							x.n.D.getIndexLength(1);
		sizeVector[2] = this->n.D.getIndexLength(2)*
							x.n.D.getIndexLength(2);
		hadaTensor.n.D = Tensor(sizeVector, 3);
		B_this = Matrix(this->n.D.getIndexLength(0)*this->n.D.getIndexLength(1),
							this->n.D.getIndexLength(2));
		B_x = Matrix(x.n.D.getIndexLength(0)*x.n.D.getIndexLength(1),
							x.n.D.getIndexLength(2));
		B_this.components = this->n.D.components;
		B_x.components = x.n.D.components;
		B_this = B_this.kron(B_x);
		hadaTensor.n.D.components = B_this.components;
	}else{
		sizeVector.resize(2);
		multiIndex.resize(2);
		sizeVector[0] = this->n.D.getIndexLength(0);
		sizeVector[1] = this->n.D.getIndexLength(1)*x.n.D.getIndexLength(1);
		hadaTensor.n.D = Tensor(sizeVector, 2);
		U_this = this->n.D;
		U_x = x.n.D;
		for(i = 0; i < sizeVector[0]; i++){
			multiIndex[0] = i;
			row_this = U_this.getRow(i);
			row_x = U_x.getRow(i);
			row_hada = row_this.kron(row_x);
			for(j = 0; j < sizeVector[1]; j++){
				multiIndex[1] = j;
				hadaTensor.n.D(multiIndex) = row_hada(j);
			}
		}
		//((Matrix*)(&hadaTensor.n.D))->print();
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return hadaTensor;
}

Tensor HTuckerMPI::full(){
	Tensor T, R;
	Matrix U_t, U_l, U_r;
	std::valarray<int> sendIntBuff;
	std::valarray<int> recvIntBuff;
	std::valarray<int> sendIndexBuff;
	std::valarray<int> recvIndexBuff;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	MPI_Status stat;
	MPI_Barrier(MPI_COMM_WORLD);
	int tag = 1;
	if(this->n.t_id == 0){
		R = Tensor(this->indexLength,
					(lapack_int)this->indexLength.size());
		recvLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		U_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		U_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag,
					MPI_COMM_WORLD, &stat);
		U_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		U_r.components = recvDoubleBuff;
		//Now compute the 0 and 1 mode product of the child matrices with
		//this node's transfer tensor
		T = U_l.modalProd(0, this->n.D);
		T = U_r.modalProd(1, T);
		//for(i = 0; i< T.getNumComponents(); i++){
//			printf("\nT.components[i] = %lf\n", T.components[i]);
	//	}
		R.components = T.components;
	}else if(this->n.index.size() == 1){
		//The Tensor stored at this node is the Matrix we need to send to the
		//parent.
		//First we resize the long buffer to send the  information about the
		//size of the matrix we are sending
		sendLongBuff.resize(2);
		sendLongBuff[0] = this->n.D.getIndexLength(0);
		sendLongBuff[1] = this->n.D.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(this->n.D.getNumComponents());
		sendDoubleBuff = this->n.D.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else{
		//In order to compute the tensor we are sending to the parent, we
		//first nneed to recieve the matrices corresponding to the left and
		//right leaves of this node.
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		U_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		U_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		U_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		U_r.components = recvDoubleBuff;
		//Now compute the 0 and 1 mode product of the child matrices with
		//this node's transfer tensor
		T = U_l.modalProd(0, this->n.D);
		T = U_r.modalProd(1, T);
		//Matricize the tensor T
		//U_t = Matrix(T.getIndexLength(0)*T.getIndexLength(1),
		//				T.getIndexLength(2));
		//U_t.components = T.components;
		//Now load the dimensions of U_t into the send buffer
		sendLongBuff[0] = T.getIndexLength(0)*T.getIndexLength(1);
		sendLongBuff[1] = T.getIndexLength(2);
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		sendDoubleBuff = T.components;
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);		
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return R;
}

HTuckerMPI& HTuckerMPI::operator=(const HTuckerMPI& E){
	this->n = E.n;
	this->indexLength = E.indexLength;
	return *this;
}

HTuckerMPI HTuckerMPI::operator+(const HTuckerMPI& B) const{
	HTuckerMPI sumTensor = *this;
	if(sumTensor.n.index.size() == 1){
		sumTensor.n.D = ((Matrix*)(&sumTensor.n.D))->cat(
							*((Matrix*)&(B.n.D)));
	}else{
		sumTensor.n.D = sumTensor.n.D.diagCat(B.n.D);
	}
	return sumTensor;
}

HTuckerMPI HTuckerMPI::operator-(const HTuckerMPI& B) const{
	HTuckerMPI difference;
	difference = B;
	if(this->getThreadID() == 0){
		difference.n.D = difference.n.D*-1;
	}
	difference = this->operator+(difference);
	return difference;
}

void HTuckerMPI::orthog(){
	Matrix T_mat, R_t, Q_t, R_l, R_r;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	std::valarray<lapack_int> sizeVector;
	MPI_Status stat;
	MPI_Barrier(MPI_COMM_WORLD);
	int tag = 1;
//	this->print();
	//establish cases for computing the different QR factorizations
	if(this->n.index.size() == 1){ //If this is a  leaf
		//Since this is a leaf, the tensor stored here is a matrix
		//We have to take the QR factorization of this matrix
		((Matrix*)&(this->n.D))->qr(Q_t, R_t);
		//Now we have to set the matrix stored here to the orthogonalized
		//matrix
		this->n.D = Q_t;
		//Now we have to send the R_t matrix to the matrix node.
		sendLongBuff.resize(2);
		sendLongBuff[0] = R_t.getIndexLength(0);
		sendLongBuff[1] = R_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(R_t.getNumComponents());
		sendDoubleBuff = R_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else if(this->n.t_id == 0){//If this is the root node
		//The first thing we must do is recieve the 2 R_l and R_r matrices
		//from the child nodes
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		R_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		R_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		R_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		R_r.components = recvDoubleBuff;
		//Now we use the two R matrices to adjust the components of B_t
		//to refect the orthogonalization of the child node
		this->n.D = R_l.modalProd(0, this->n.D);
		this->n.D = R_r.modalProd(1, this->n.D);
	}else{//Last case, must be an interior node
		//The first thing we must do is recieve the 2 R_l and R_r matrices
		//from the child nodes
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		R_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		R_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		R_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		R_r.components = recvDoubleBuff;
		//Now we use the two R matrices to adjust the components of B_t
		//to refect the orthogonalization of the child node
		this->n.D = R_l.modalProd(0, this->n.D);
		this->n.D = R_r.modalProd(1, this->n.D);
		//Now we matricize into a temp matrix and compute a QR decomp
		T_mat = Matrix(this->n.D.getIndexLength(0)*this->n.D.getIndexLength(1),
									this->n.D.getIndexLength(2));
		T_mat.components = this->n.D.components;
		T_mat.qr(Q_t, R_t);
		//Now set the transfer tensor's components into this tensor
		sizeVector.resize(3);
		sizeVector[0] = this->n.D.getIndexLength(0);
		sizeVector[1] = this->n.D.getIndexLength(1);
		sizeVector[2] = Q_t.getNumCols();
		this->n.D = Tensor(sizeVector, 3);
		this->n.D.components = Q_t.components;
		//Finally, send the R_t matrices up to the parent node
		sendLongBuff.resize(2);
		sendLongBuff[0] = R_t.getIndexLength(0);
		sendLongBuff[1] = R_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(R_t.getNumComponents());
		sendDoubleBuff = R_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

Matrix HTuckerMPI::gramians(){
	Matrix T_mat, M_t, M_l, M_r, G_t, G_l, G_r;
	Tensor T;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	MPI_Status stat;
	std::valarray<lapack_int> matricizeIndex(2);
	MPI_Barrier(MPI_COMM_WORLD);
	int tag = 1;
	//Separate into child and non-child noes
	if(this->n.index.size() == 1){
		//At the leaves, we store the basis matrices.
		//We cast the address of the tensor to a Matrix* and then access
		//the transpose and muliplication functions
		M_t = ((Matrix*)&(this->n.D))->transp().operator*(
							*((Matrix*)&(this->n.D)));
		//Now send the matrix to the parent node
		sendLongBuff.resize(2);
		sendLongBuff[0] = M_t.getIndexLength(0);
		sendLongBuff[1] = M_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(M_t.getNumComponents());
		sendDoubleBuff = M_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}else if(this->n.t_id == 0){
		//Recieve the M_l and M_r matrices from the children
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		M_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		M_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_r.components = recvDoubleBuff;
		//Then set the G_t matrix to contain only the value 1
		G_t = Matrix(1, 1);
		G_t(0,0) = 1;
	}else{//This is an interior node
		//Recieve the M_l and M_r matrices from the children
		recvLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		M_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		M_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		M_r.components = recvDoubleBuff;
		//Now matricize this transfer tensor
		matricizeIndex[0] = 0;
		matricizeIndex[1] = 1;
		T_mat = Matrix(this->n.D, matricizeIndex, 2);
		T_mat = T_mat.transp();
		//Then compute M_t
		//M_t = T_mat.transp()*(M_r.kron(M_l))*T_mat;
//printf("\ntask %d : this->n.D is %ldx%ldx%ld\n",
//	this->n.t_id,this->n.D.getIndexLength(0),this->n.D.getIndexLength(1),this->n.D.getIndexLength(2));
		T = M_r.modalProd(1,this->n.D);
//printf("\ntask %d : T is %ldx%ldx%ld\n",
//	this->n.t_id,T.getIndexLength(0),T.getIndexLength(1),T.getIndexLength(2));
		T = M_l.modalProd(0,T);
		matricizeIndex[0] = 0;
		matricizeIndex[1] = 1;
		M_t =  Matrix(T, matricizeIndex, 2);
		M_t = T_mat*M_t;
		//Finally send it to the parent
		sendLongBuff.resize(2);
		sendLongBuff[0] = M_t.getIndexLength(0);
		sendLongBuff[1] = M_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(M_t.getNumComponents());
		sendDoubleBuff = M_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);//PROBABLY NOT NEEDED?
	//Now that each node contains the needed M_r and M_l, we compute
	//the gramians and pass back to the children
	if(this->n.index.size() == 1){
		//The parent node has computed the gramian for this node.
		//All we need to do is recieve it.
		recvLongBuff.resize(2);
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD, &stat);
		G_t = Matrix(recvLongBuff[0], recvLongBuff[1]);
		G_t.components = recvDoubleBuff;
	}else if(this->n.t_id == 0){
		//In this case, we have both M_l and M_r, we need to compute
		//the gramian corresponding to each and then send to the child.
		T_mat = this->n.D;
		G_r =  (T_mat.transp()) * M_l * T_mat ;
		G_l =  T_mat * M_r *  (T_mat.transp());
		//Now send G_r and G_l
		sendLongBuff.resize(2);
		sendLongBuff[0] = G_l.getIndexLength(0);
		sendLongBuff[1] = G_l.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(G_l.getNumComponents());
		sendDoubleBuff = G_l.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the left node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD);
		sendLongBuff[0] = G_r.getIndexLength(0);
		sendLongBuff[1] = G_r.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(G_r.getNumComponents());
		sendDoubleBuff = G_r.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the right node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD);
	}else{//Interior node
		//For this case, we need to compute the gramians for the
		//child nodes and also recieve this node's gramian
		//We'll recieve, compute, then send
		//First recieve from parent
		recvLongBuff.resize(2);
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD, &stat);
		G_t = Matrix(recvLongBuff[0], recvLongBuff[1]);
		G_t.components = recvDoubleBuff;
		//matricize this transfer tensor
		T = G_t.modalProd(2,this->n.D);
		T = M_r.modalProd(1,T);
		matricizeIndex[0] = 1;
		matricizeIndex[1] = 2;
		G_l = Matrix(T, matricizeIndex,2);
		matricizeIndex[0] = 0;
		T_mat = Matrix(this->n.D, matricizeIndex, 1);
		//printf("\nHere lies err?\n");
		G_l = T_mat * G_l;
		//now compute right gramian
		T = G_t.modalProd(2, this->n.D);
		T = M_l.modalProd(0,T);
		matricizeIndex[0] = 0;
		matricizeIndex[1] = 2;
		G_r = Matrix(T, matricizeIndex, 2);
		matricizeIndex[0] = 1;
		T_mat = Matrix(this->n.D, matricizeIndex, 1);
		//printf("\nOr maybe here lies err?\n");
		G_r = T_mat * G_r;
		//OK, now send to children
		sendLongBuff.resize(2);
		sendLongBuff[0] = G_l.getIndexLength(0);
		sendLongBuff[1] = G_l.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(G_l.getNumComponents());
		sendDoubleBuff = G_l.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the left node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD);
		sendLongBuff[0] = G_r.getIndexLength(0);
		sendLongBuff[1] = G_r.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(G_r.getNumComponents());
		sendDoubleBuff = G_r.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the right node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return G_t;
}

void HTuckerMPI::truncateHT(lapack_int maxRank){
	Matrix S_t, S_l, S_r, B_t, G_t;
	std::valarray<long> sendLongBuff;
	std::valarray<long> recvLongBuff;
	std::valarray<double> sendDoubleBuff;
	std::valarray<double> recvDoubleBuff;
	MPI_Status stat;
	int tag=1;
	std::valarray<lapack_int> matricizeIndex(2);
	MPI_Barrier(MPI_COMM_WORLD);
	this->orthog();
	MPI_Barrier(MPI_COMM_WORLD);
	G_t = this->gramians();
	MPI_Barrier(MPI_COMM_WORLD);
	//First compute the left SVs to truncate
	if(this->n.t_id != 0){ //If this is not the root node
		//printf("\n(m,n) = (%ld,%ld)\tnode %d\n",G_t.getNumRows
				//(),G_t.getNumCols(), this->n.t_id);
		S_t = G_t.getLeftSV(maxRank);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//this->print();
	//Now separate into cases for the hierarchical truncation
	if(this->n.index.size() == 1){//If this is a leaf
		//Now pass the elements of of S_t up to the parent
		sendLongBuff.resize(2);
		sendLongBuff[0] = S_t.getIndexLength(0);
		sendLongBuff[1] = S_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(S_t.getNumComponents());
		sendDoubleBuff = S_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
		//Then we multiply the U_t leaf by the truncated matrix
		this->n.D = ((Matrix*)&(this->n.D))->operator*(S_t);
	}else if(this->n.t_id != 0){//if this is an interior node
		//First we recieve the S_r and S_l messages from the child, then send
		//the S_t message to the parent, then compute
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		S_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		S_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		S_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		S_r.components = recvDoubleBuff;
		//S_t.print();
		sendLongBuff[0] = S_t.getIndexLength(0);
		sendLongBuff[1] = S_t.getIndexLength(1);
		//Now load the components of this node's matrix into the double buffer
		sendDoubleBuff.resize(S_t.getNumComponents());
		sendDoubleBuff = S_t.components;
		//Send the memory size as an MPI message
		MPI_Send(	&(sendLongBuff[0]),
					2, MPI_LONG, this->n.p_id, tag, MPI_COMM_WORLD);
		//Now send the double data to the parent node
		MPI_Send(	&(sendDoubleBuff[0]),
					(int) sendDoubleBuff.size(),
					MPI_DOUBLE, this->n.p_id, tag, MPI_COMM_WORLD);
		this->n.D = S_l.transp().modalProd(0, this->n.D);
		this->n.D = S_r.transp().modalProd(1, this->n.D);
		matricizeIndex[0] = 0;
		matricizeIndex[1] = 1;
		B_t = Matrix(this->n.D, matricizeIndex, 2);
		B_t = B_t * S_t;
		matricizeIndex.resize(3);
		matricizeIndex[0] = this->n.D.getIndexLength(0);
		matricizeIndex[1] = this->n.D.getIndexLength(1);
		matricizeIndex[2] = B_t.getNumCols();
		this->n.D = Tensor(matricizeIndex, 3);
		this->n.D.components = B_t.components;
	}else{//otherwise it is root
		recvLongBuff.resize(2);
		sendLongBuff.resize(2);
		//First recieve from left child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.l_id, tag, MPI_COMM_WORLD, &stat);
		S_l = Matrix(recvLongBuff[0], recvLongBuff[1]);
		S_l.components = recvDoubleBuff;
		//Second recieve from right child
		MPI_Recv(	&(recvLongBuff[0]),
					2, MPI_LONG, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		recvDoubleBuff.resize(recvLongBuff[0] * recvLongBuff[1]);
		MPI_Recv(	&(recvDoubleBuff[0]),
					(int) recvDoubleBuff.size(),
					MPI_DOUBLE, this->n.r_id, tag, MPI_COMM_WORLD, &stat);
		S_r = Matrix(recvLongBuff[0], recvLongBuff[1]);
		S_r.components = recvDoubleBuff;
		//printf("\ntaskID %d\n", this->n.t_id);
		//S_l.print();
		//S_r.print();
		this->n.D = S_l.transp().modalProd(0, this->n.D);
		this->n.D = S_r.transp().modalProd(1, this->n.D);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void HTuckerMPI::scale(double a){
	this->n.D = this->n.D * a;
}

void HTuckerMPI::leafMult(Matrix&  L){
	if(this->n.index.size() == 1){
		Matrix *U_t = ((Matrix*)&this->n.D);
		this->n.D = L.operator*(*U_t);
	}
}

HTuckerMPI HTuckerMPI::linOpApply(HTuckerMPI& x){
	HTuckerMPI p = x;
	Matrix thisKron;
	Matrix xKron;
	Matrix Tmat;
	Matrix Uvec;
	Matrix Vvec;
	lapack_int i, r, l, k, thisNumCols, xNumCols, vNumRows;
	std::valarray<lapack_int> sizeVector;
	if(x.n.index.size() == 1){
		//Check if *this as linear operator is valid interpretation
		if(this->n.D.getIndexLength(0) % x.n.D.getIndexLength(0) != 0){
			printf("\nHTucker LinOpApply Error:\n"
					"\nargument's number of rows mismatches this linear"
					"operator's number of columns.\n");
			exit(1);
		}
		//Start forming operator matrices
		thisNumCols = this->n.D.getIndexLength(1);
		xNumCols = x.n.D.getIndexLength(1);
		Tmat = Matrix(this->n.D.getIndexLength(0) / x.n.D.getIndexLength(0),
						x.n.D.getIndexLength(0));
		p.n.D = Matrix(Tmat.getNumRows(), thisNumCols*xNumCols);
		//Loop to compute application of lienar operator
		for(r = 0; r < thisNumCols; r++){
			for(l = 0; l < xNumCols; l++){
				k = r + (thisNumCols*l);
				Uvec = ((Matrix*)&(x.n.D))->getCol(l);
				Tmat.components = (((Matrix*)&(x.n.D))->getCol(l)).components;
				//Call a lapack routine to do the operation
				Vvec = Tmat*Uvec;
				vNumRows = Vvec.getNumRows();
				for(i = 0; i < vNumRows; i++){
					((Matrix*)&(p.n.D))->entry(i,k) = Vvec(i,0);
				}
			}
		}
	}else{
		//Start doing the kronecker product. first allocation memory for it
		sizeVector.resize(3);
		sizeVector[0] = this->n.D.getIndexLength(0)*x.n.D.getIndexLength(0);
		sizeVector[1] = this->n.D.getIndexLength(1)*x.n.D.getIndexLength(1);
		sizeVector[2] = this->n.D.getIndexLength(2)*x.n.D.getIndexLength(2);
		p.n.D = Tensor(sizeVector, 3);
		thisKron = Matrix(this->n.D.getIndexLength(0)*
							this->n.D.getIndexLength(1),
							this->n.D.getIndexLength(2));
		xKron = Matrix(x.n.D.getIndexLength(0)*
							x.n.D.getIndexLength(1),
							x.n.D.getIndexLength(2));
		//set the components of the arrays
		thisKron.components = this->n.D.components;
		xKron.components = x.n.D.components;
		//actuall compute kronecker
		p.n.D.components = (thisKron.kron(xKron)).components;
	}
	//done, return.
	return p;
}

int HTuckerMPI::getThreadID() const{
	return this->n.t_id;
}

lapack_int HTuckerMPI::getTreeIndex(int i) const{
	return this->n.index[i];
}

int HTuckerMPI::getTreeIndexSize() const{
	return ((int) this->n.index.size());
}

void HTuckerMPI::print() const{
	MPI_Barrier(MPI_COMM_WORLD);
	lapack_int i;
	long blankMsg = 0;
	MPI_Status stat;
	int tag=1, numCPUs;
	MPI_Comm_size(MPI_COMM_WORLD, &numCPUs);
	if(this->n.t_id == 0){
		this->n.printID();
		for(i = 0; i < (lapack_int)this->n.D.getDim() - 1; i++){
			printf("%ld ", this->n.D.getIndexLength(i));
		}
		printf("%ld\n", this->n.D.getIndexLength(i));
		printf("%ld\n", (long)this->n.D.components.size());
		MPI_Send(	&(blankMsg),
					1, MPI_LONG, this->n.t_id+1, tag, MPI_COMM_WORLD);
	}else{
		MPI_Recv(	&(blankMsg),
					1,
					MPI_LONG, this->n.t_id-1, tag, MPI_COMM_WORLD, &stat);
		this->n.printID();
		for(i = 0; i < (lapack_int)this->n.D.getDim() - 1; i++){
			printf("%ld ", this->n.D.getIndexLength(i));
		}
		printf("%ld\n", this->n.D.getIndexLength(i));
		printf("%ld\n", (long)this->n.D.components.size());
		if(this->n.t_id < numCPUs-1){
			MPI_Send(	&(blankMsg),
				1, MPI_LONG, this->n.t_id+1, tag, MPI_COMM_WORLD);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

