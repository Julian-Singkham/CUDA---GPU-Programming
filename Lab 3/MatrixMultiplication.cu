/**
  ***********************************************************************************
  * @file   : MatrixMultiplication.cu
  * @brief  : Main program body
  *         : Lab 3: Basic Matrix Multiplication
  *         : CS-4981/031 
  * @date   : SEP 29 2021
  * @author : Julian Singkham
  ***********************************************************************************
  * @attention
  *
  * This program conducts matrix multiplication with dimensions specified by the user
  * in the program arguments and each element is set to 1. 
  *
  * Inputs:
  *    m = # rows in matrix A
  *    n = # columns in matrix A and rows in matrix B
  *    k = # columns in matrix B
  *
  * Part 1:
  * Matrix A and B are allocated in memory and set to 1. Then the CPU conducts matrix
  * multiplicaiton and the output is verified. The output matrix is set to 0 and the
  * CUDA implementation copies matrix A and B to global memory before conducting
  * Kernel multiplication. After the kernel the output is veridied. In both 
  * implementations, only the multiplicaiton is timed.
  *
  * Part 2:
  * The CPU implementation is the same except the creation and initialization of
  * the matrices are included in the timing. The GPU implementation includes
  * 2 additional kernels that create and set the value of matrix A and B in global
  * memory. The GPU is timed for the creation of matrix A and B, mulitplication, 
  * and copying the output matrix to the host.
  *
  ***********************************************************************************
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//====================================CUDA Kernel=====================================
/**
  * @brief Multiplies two vectors to an output vector
  * 
  * @param output: Pointer to the the device vector that stores the multiplicaiton.
  * @param mata: Pointer to the the device vector that holds matrix A.
  * @param matb: Pointer to the the device vector that holds matrix B.
  * @param m: Number of rows in matrix A.
  * @param n: Number of cols in matrix A and rows in matrix B.
  * @param k: Number of cols in matrix B.
  * 
  * @retval NONE
  */
__global__ void VectorMulKernel(int* output, int* mata, int* matb, size_t m, size_t n, size_t k){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	double sum = 0.0;
	// Each thread is a row (matrix A) column (matrix B) pair
	if(col < k && row < m){
		// Each loop is a row column operation
		for (int i = 0; i < n; i++)
			sum += mata[row*n+i] * matb[i*k+col];
		output[row*k+col] = sum;
	}
}

/**
  * @brief Initalizes matrix A to 1
  * 
  * @param mata: Pointer to the the device vector that holds matrix A.
  * @param m: Number of rows in matrix A.
  * @param n: Number of cols in matrix A.
  * 
  * @retval NONE
  */
  __global__ void VectorMatAInitialize(int* mata, size_t m, size_t n){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread is an element of the matrix
	if(col < m*n)
      	mata[col] = 1;
}

/**
  * @brief Initalizes matrix B to 1
  * 
  * @param matb: Pointer to the the device vector that holds matrix B.
  * @param n: Number of rows in matrix B.
  * @param k: Number of cols in matrix B.
  * 
  * @retval NONE
  */
  __global__ void VectorMatBInitialize(int* matb, size_t n, size_t k){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread is an element of the matrix
	if(col < n*k)
      	matb[col] = 1;
}

//======================================Methods=======================================
/**
  * @brief Calculates how long the CPU was operating
  * 
  * @param start: Pointer to the cpu time the operation started.
  * @param end: Pointer to the cpu time the operation ended.
  * 
  * @retval The length of time the cpu was operating in milliseconds
  */
float cpu_time(timespec* start, timespec* end){
	return ((1e9*end->tv_sec + end->tv_nsec) - 
			(1e9*start->tv_sec + start->tv_nsec))/1e6;
}

/**
  * @brief Multiplies two matrices to an output matrix.
  * 
  * @param matc: Pointer to the the device vector that stores the multiplicaiton.
  * @param mata: Pointer to the the device vector that holds matrix A.
  * @param matb: Pointer to the the device vector that holds matrix B.
  * @param m: Number of rows in matrix A.
  * @param n: Number of cols in matrix A and rows in matrix B.
  * @param k: Number of cols in matrix B.
  *
  * @retval NONE
  */
void mat_Mul(int* matc, int* mata, int* matb, size_t m, size_t n, size_t k){
	for (int rowa = 0; rowa < m; rowa++)
		for (int colb = 0; colb < k; colb++)
			for (int cola = 0; cola < n; cola++)
				matc[rowa*k+colb] += mata[rowa*n+cola] * matb[cola*k+colb];
}


/**
  * @brief Conducts matrix multiplication between two matrices and saves the
  * output to another matrix. This also compares the compute time between 
  * the CPU and GPU
  * 
  * @param argv: Pointer to the given arguements
  *     args[1] = Matrix A row count
  *     args[2] = Matrix A col/Matrix B row count
  *     args[3] = Matrix B col count
  * 
  * @retval NONE
  */
int main(int argc, char* argv[]){
	printf("**************Initializing CPU Variables**************\n");
	size_t m = atoi(argv[1]);
	size_t n = atoi(argv[2]);
	size_t k = atoi(argv[3]);
	size_t mata_size = m * n * sizeof(int);
	size_t matb_size = n * k * sizeof(int);
	size_t matc_size = m * k * sizeof(int);
	timespec ts, te; // Used to calculate CPU implementation

	clock_gettime(CLOCK_MONOTONIC_RAW, &ts); // Use in part 2
	// Allocate memory on the host
	int*mata = (int*)malloc(mata_size);
	int*matb = (int*)malloc(matb_size);
	int*matc = (int*)calloc(m*k, sizeof(int));

	// Create matrix A
    for (int i = 0; i < m * n; i++)
      	mata[i] = 1.0;

	// Create matrix B
    for (int i = 0; i < n * k; i++)
		matb[i] = 1.0;
	
	printf("******************CPU Implementation******************\n");
	//clock_gettime(CLOCK_MONOTONIC_RAW, &ts); // Use in part 1
	mat_Mul(matc, mata, matb, m, n, k);
	clock_gettime(CLOCK_MONOTONIC_RAW, &te);

	//Validate matrix C
	double matc_value = 0.0;
	for(int i=0; i<m; i++)
		for(int j=0; j<k; j++)
			matc_value += matc[i*k+j];
	matc_value = matc_value/(m*k);
	double difference = matc_value - n;

	printf("Calculated Output = %lf\n", matc_value);
	printf("True Output = %d\n", n);
	printf("Output Difference = %lf\n", difference);
	printf("\nCPU ellapsed time: %f milliseconds\n\n", cpu_time(&ts, &te));


	printf("****************Clearing Output Memory****************\n");
	memset(matc, 0, m*k*sizeof(int));
	matc_value = 0.0;
	for(int i=0; i<m; i++)
		for(int j=0; j<k; j++)
			matc_value += matc[i*k+j];
	printf("Sum should be 0. Sum in memory = %lf\n\n", matc_value);


	printf("**************Initializing GPU Variables**************\n");
	int* d_mata;
	int* d_matb;
	int* d_matc;

	// Create CUDA start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory on the GPU
	cudaEventRecord(start); // Use in part 2
	cudaMalloc((void **)&d_mata, mata_size);
	cudaMalloc((void **)&d_matb, matb_size);
	cudaMalloc((void **)&d_matc, matc_size);
	
	/* Use in part 1
	// Copy host variables to GPU
	cudaMemcpy(d_mata, mata, mata_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matb, matb, matb_size, cudaMemcpyHostToDevice);
	*/ // Use in part 1

	///* // Use in part 2
	VectorMatAInitialize<<<ceil(m*n/1024.0), 1024>>>(d_mata, m, n); // Create matrix A
	// Calculate grid size and block dimension
	VectorMatBInitialize<<<ceil(n*k/1024.0), 1024>>>(d_matb, n, k); // Create matrix B
	///* // Use in part 2
	

	printf("******************GPU Implementation******************");
	// Calculate grid size and block dimension
	unsigned int grid_rows = (m+32-1)/32;
    unsigned int grid_cols = (k+32-1)/32;
    dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(32, 32);

	//cudaEventRecord(start); // Use in part 1
	VectorMulKernel<<<dimGrid, dimBlock>>>(d_matc, d_mata, d_matb, m, n, k);
	cudaMemcpy(matc, d_matc, matc_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(mata, d_mata, mata_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0; //declare a variable to store runtime
 	cudaEventElapsedTime(&milliseconds, start, stop); //get the elapsed time

	// Validate matrix C
	matc_value = 0.0;
	for(int i=0; i<m; i++)
		for(int j=0; j<k; j++)
			matc_value += matc[i*k+j];
	matc_value = matc_value/(m*k);
	difference = matc_value - n;

	printf("\nCalculated Output = %lf\n", matc_value);
	printf("True Output = %d\n", n);
	printf("Output Difference = %lf\n", difference);
	printf("\nCPU ellapsed time: %f milliseconds\n\n", milliseconds);
	
	// Free all host allocated memory
	free(mata);
	free(matb);
	free(matc);

	// Free all GPU allocated memory
	cudaFree(d_mata);
	cudaFree(d_matb);
	cudaFree(d_matc);
	return 0;
}





