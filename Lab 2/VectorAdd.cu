/**
  ***********************************************************************************
  * @file   : VectorAdd.cu
  * @brief  : Main program body
  *         : Lab 2: CUDA Vector Add
  *         : CS-4981/031 
  * @date   : SEP 21 2021
  * @author : Julian Singkham
  ***********************************************************************************
  * @attention
  *
  * This program conducts vector addition with a vector specified by the user as
  * an argument in the program call. The program creates three vectors of the size
  * specified by the user and conducts the following operation:
  *
  * c[i] = a[i] + b[i]
  * Where the sum of c[]/c.size should equal 1
  * a[i] = sin(i)^2 and b[i] = cos(i)^2
  *
  * This program compares the timing between the CPU and GPU when conducting the 
  * vector addition.
  *
  ***********************************************************************************
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//====================================CUDA Kernel=====================================
/**
  * @brief Adds two vectors together and saves the output to a specified vector.
  * 
  * @param output: Pointer to the the device vector that stores the sum.
  * @param sint: Pointer to the the device vector that holds the sin^2 vector.
  * @param cost: Pointer to the the device vector that holds the cos^2 vector.
  * @param size: Number of elements in the vectors.  
  * 
  * @retval NONE
  */
__global__ void VectorAddKernel(float* output, float* sint, float* cost, int size){
	int col = blockIdx.x * blockDim.x + threadIdx.x; //Thread infexer
	if(col < size)
		output[col] = sint[col] + cost[col];
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
  * @brief Conducts vector addition between two vectors and saves the output to
  * another vector. This also compares the compute time between the CPU and GPU
  * 
  * @param argv: Pointer to the given arguements
  *     args[1] = Size of the vector
  * 
  * @retval NONE
  */
int main(int argc, char* argv[]){
	int input = atoi(argv[1]);
	printf("***Initializing Data With Vector Size of %d Elements***\n", input);
	float* sint = (float *) calloc(input, sizeof(float));
	float* cost = (float *) calloc(input, sizeof(float));
	float* output = (float *) calloc(input, sizeof(float));

	for(int i=0; i<input; i++){
		sint[i] = float(sin(i) * sin(i));
		cost[i] = float(cos(i) * cos(i));
	}

	printf("******************CPU Implementation******************\n");
	// Used to calculate CPU implementation
	timespec ts, te;

	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	for(int i=0; i<input; i++){
		output[i] = sint[i] + cost[i];
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &te);

	double sum = 0.0;
	for(int i=0; i<input; i++)
		sum += output[i];
	sum = sum/input;
	double difference = sum - 1.0;

	printf("Calculated Output = %lf\n", sum);
	printf("True Output = 1\n");
	printf("Output Difference = %lf\n", difference);
	printf("\nCPU ellapsed time: %f milliseconds\n\n", cpu_time(&ts, &te));

	printf("****************Clearing Output Memory****************\n");
	// Clear output memory
	memset(output, 0, input*sizeof(float));
	sum = 0.0;
	for(int i=0; i<input; i++)
		sum += output[i];
	printf("Sum should be 0. Sum in memory = %lf\n\n", sum);


	printf("**************Initializing GPU Variables**************\n");
	float* d_sint;
	float* d_cost;
	float* d_output;
	int size = input*sizeof(float);

	// Allocate memory on the GPU
	cudaMalloc((void **)&d_sint, size);
	cudaMalloc((void **)&d_cost, size);
	cudaMalloc((void **)&d_output, size);
	
	// Create CUDA start and stop events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Copy host variables to GPU
	cudaMemcpy(d_sint, sint, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cost, cost, size, cudaMemcpyHostToDevice);
	
	printf("******************GPU Implementation******************\n");
	cudaEventRecord(start);
	VectorAddKernel<<<ceil(size/32), 32>>>(d_output, d_sint, d_cost, input);
	cudaEventRecord(stop);
	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0; //declare a variable to store runtime
 	cudaEventElapsedTime(&milliseconds, start, stop); //get the elapsed time
	
	sum = 0.0;
	for(int i=0; i<input; i++)
		sum += output[i];
    sum = sum/input;
	difference = sum - 1.0;

	printf("Calculated Output = %lf\n", sum);
	printf("True Output = 1\n");
	printf("Output Difference = %lf\n", difference);
	printf("\nCPU ellapsed time: %f milliseconds\n\n", milliseconds);
	
	// Free all CPU allocated memory
	free(sint);
	free(cost);
	free(output);

	// Free all GPU allocated memory
	cudaFree(d_sint);
	cudaFree(d_cost);
	cudaFree(d_output);
	return 0;
}





