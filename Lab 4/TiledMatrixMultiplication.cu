/**
  ***********************************************************************************
  * @file   : TiledMatrixMultiplication.cu
  * @brief  : Main program body
  *         : Lab 4: Tiled Matrix Multiplication
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
  * Matrix A and B are created and initialized to 1 in the kernel. The GPU
  * implementation conducts matrix multiplication in global memory. The tiled
  * version conducts matrix multiplicatio using shared memory. Between the two
  * implementations the output matrix is zeroed. In both implementations, only the 
  * multiplicaiton kernel call is timed.
  *
  ***********************************************************************************
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const int TILE_WIDTH = 32;
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
  * @brief Multiplies two vectors to an output vector using shared memory
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
  __global__ void VectorTiledMulKernel(int* output, int* mata, int* matb, size_t m, size_t n, size_t k){
	__shared__ int shareMatA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int shareMatB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; 
	int by = blockIdx.y;
    int tx = threadIdx.x; 
	int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float temp = 0;
    for(int i = 0; i < ceil((float) n/TILE_WIDTH); ++i){
		// Must check for boundary condition in each memory call
		if(row < m && i * TILE_WIDTH+tx < n)
			shareMatA[ty][tx] = mata[row*n + (i*TILE_WIDTH + tx)];
		else
			shareMatA[ty][tx] = 0;
			
		if (i*TILE_WIDTH+ty < n && col < k)
    		shareMatB[ty][tx] = matb[(i*TILE_WIDTH + ty)*k + col];
		else
			shareMatB[ty][tx] = 0;
    	__syncthreads();
		
		if(row < m && col < k)
      		for(int j = 0; j < TILE_WIDTH; ++j)
      			temp += shareMatA[ty][j] * shareMatB[j][tx];
		__syncthreads();
    }
	if (row < m && col < k)
    	output[row*k + col] = temp;
}

/**
  * @brief Initalizes matrix to 1
  * 
  * @param mata: Pointer to the the device vector that holds the matrix.
  * @param m: Number of rows in the matrix.
  * @param n: Number of cols in the matrix.
  * 
  * @retval NONE
  */
  __global__ void VectorMatrixInitialize(int* mat, size_t m, size_t n){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread is an element of the matrix
	if(col < m*n)
      	mat[col] = 1;
}

//======================================Methods=======================================
/**
  * @brief Conducts matrix multiplication between two matrices and saves the output
  * to another matrix. This also compares the compute time between the GPU and GPU 
  * with tiled multiplication.
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

	// Allocate memory on the host
	int*matc = (int*)calloc(m*k, sizeof(int));

	printf("**************Initializing GPU Variables**************\n");
	int* d_mata;
	int* d_matb;
	int* d_matc;

	// Create CUDA start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory on the GPU
	cudaMalloc((void **)&d_mata, mata_size);
	cudaMalloc((void **)&d_matb, matb_size);
	cudaMalloc((void **)&d_matc, matc_size);

	// Calculate grid size and block dimension
	unsigned int grid_rows = ceil((float) m/TILE_WIDTH);
    unsigned int grid_cols = ceil((float) k/TILE_WIDTH);
    dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	VectorMatrixInitialize<<<ceil(m*n/1024.0), 1024>>>(d_mata, m, n); // Create matrix A
	VectorMatrixInitialize<<<ceil(n*k/1024.0), 1024>>>(d_matb, n, k); // Create matrix B
	

	printf("******************GPU Implementation******************");
	cudaEventRecord(start);
	VectorMulKernel<<<dimGrid, dimBlock>>>(d_matc, d_mata, d_matb, m, n, k);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(matc, d_matc, matc_size, cudaMemcpyDeviceToHost);
	float milliseconds = 0; //declare a variable to store runtime
 	cudaEventElapsedTime(&milliseconds, start, stop); //get the elapsed time

	// Validate matrix C
	double matc_value = 0.0;
	for(int i=0; i<m; i++)
		for(int j=0; j<k; j++)
			matc_value += matc[i*k+j];
	matc_value = matc_value/(m*k);
	double difference = matc_value - n;

	printf("\nCalculated Output = %lf\n", matc_value);
	printf("True Output = %d\n", n);
	printf("Output Difference = %lf\n", difference);
	printf("\nGPU ellapsed time: %f milliseconds\n\n", milliseconds);

	cudaMemset(d_matc, 0, m*k);
	

	printf("***************GPU Tiled Implementation***************");
	cudaEventRecord(start);
	VectorTiledMulKernel<<<dimGrid, dimBlock>>>(d_matc, d_mata, d_matb, m, n, k);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(matc, d_matc, matc_size, cudaMemcpyDeviceToHost);
	milliseconds = 0; //declare a variable to store runtime
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
	printf("\nGPU ellapsed time: %f milliseconds\n\n", milliseconds);


	// Free all host allocated memory
	free(matc);

	// Free all GPU allocated memory
	cudaFree(d_mata);
	cudaFree(d_matb);
	cudaFree(d_matc);
	return 0;
}





